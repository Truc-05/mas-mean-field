from __future__ import annotations

import json
import os
import re
import threading
import time

import numpy as np

from mfg_llm.opinion_state import extract_opinion_state

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

_LABELS = "ABCDEFGHIJ"
_LABEL_WORDS = ("option", "answer", "choice", "lựa chọn", "đáp án", "phương án", "")


def _norm(s: object) -> str:
    """Lowercase, collapse whitespace, strip surrounding punctuation — used to
    match score keys against option text/labels regardless of quoting or `)`/`.`."""
    return re.sub(r"\s+", " ", str(s)).strip().lower().strip(" .)(:-")


def _resolve_key(key: str, n: int, label_idx: dict[str, int],
                 text_idx: dict[str, int], answer_space: list[str]) -> int | None:
    """Resolve one model-supplied score key to an option index, or None."""
    kn = _norm(key)
    if not kn:
        return None
    if len(kn) == 1 and kn in label_idx:                       # "A"  /  "A)"  /  "A."
        return label_idx[kn]
    m = re.match(r"^(.*?)\s*([a-j])$", kn)                      # "option a", "đáp án b"
    if m and m.group(1) in _LABEL_WORDS and m.group(2) in label_idx:
        return label_idx[m.group(2)]
    if kn in text_idx:                                          # exact option text
        return text_idx[kn]
    if kn.isdigit():                                            # 1-based or 0-based index
        j = int(kn)
        if 1 <= j <= n:
            return j - 1
        if 0 <= j < n:
            return j
    subs = [i for i, a in enumerate(answer_space)               # unique substring match
            if kn in _norm(a) or _norm(a) in kn]
    return subs[0] if len(subs) == 1 else None


class RateLimiter:
    """Thread-safe limiter spacing request starts evenly at `rpm` per minute.

    Shared across all agents in a process so parallel workers cannot burst past
    a provider's per-minute quota (Groq free tier: ~30 req/min and ~6k tokens/min
    for llama-3.1-8b-instant — in practice rpm ~20 is sustainable)."""

    def __init__(self, rpm: float):
        self.min_interval = 60.0 / float(rpm)
        self._lock = threading.Lock()
        self._next_time = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next_time - now
            self._next_time = max(now, self._next_time) + self.min_interval
        if wait > 0:
            time.sleep(wait)


_limiters: dict[float, RateLimiter] = {}
_limiters_lock = threading.Lock()


def get_rate_limiter(rpm: float) -> RateLimiter:
    """Process-wide shared limiter per rpm value (so every debate/arm/question
    draws from the same quota)."""
    with _limiters_lock:
        if rpm not in _limiters:
            _limiters[rpm] = RateLimiter(rpm)
        return _limiters[rpm]


_vertex_creds = None
_vertex_creds_lock = threading.Lock()


def get_vertex_token() -> str:
    """Return a valid Vertex AI access token from Application Default Credentials,
    refreshing it when expired. Credentials are process-wide and guarded by a lock
    (google-auth refresh is not thread-safe) so parallel workers can share one token
    and a run longer than the ~1h token lifetime doesn't 401 mid-way."""
    global _vertex_creds
    with _vertex_creds_lock:
        if _vertex_creds is None:
            import google.auth
            _vertex_creds, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        if not _vertex_creds.valid:
            from google.auth.transport.requests import Request
            _vertex_creds.refresh(Request())
        return _vertex_creds.token


class DebateAgent:
    """One debate participant backed by an LLM API.

    backend:
      - "groq":      OpenAI-compatible, free-tier chat models (llama-3.1-8b-instant,
                     llama-3.3-70b-versatile, ...). Key from GROQ_API_KEY.
      - "openai":    any OpenAI-compatible server (Ollama/vLLM/LM Studio) via base_url.
                     Key from OPENAI_API_KEY (dummy value is fine for local servers).
      - "vertex":    Vertex AI OpenAI-compatible endpoint (base_url points at
                     .../endpoints/openapi, model like google/gemini-2.5-flash-lite).
                     Auth via Application Default Credentials (gcloud auth
                     application-default login); the access token is auto-refreshed
                     per request, so long runs don't die when it expires (~1h).
      - "anthropic": Anthropic API. Key from ANTHROPIC_API_KEY.

    The model must be a *chat* model. Classifier models such as
    meta-llama/llama-prompt-guard-2-86m only emit safety labels, cannot produce
    the JSON scores this agent parses, and would silently zero out every
    opinion state.
    """

    def __init__(self, agent_id: int, model: str = "llama-3.1-8b-instant",
                 backend: str = "groq", base_url: str | None = None,
                 api_key: str | None = None, client=None,
                 temperature: float = 1.0, max_retries: int = 5,
                 max_flat_retries: int = 2, persona: str | None = None,
                 rate_limiter: RateLimiter | None = None,
                 timeout: float = 60.0):
        self.agent_id = agent_id
        self.model = model
        self.backend = backend
        # a distinct reasoning stance prepended to the prompt. Homogeneous agents
        # (same model, same prompt) make near-identical, *correlated* errors, so a
        # crowd of weak models can't beat a single one — this is the lever that
        # decorrelates them (see the diversity-injection control).
        self.persona = persona
        self.max_retries = max_retries
        # a reply that yields a flat (uninformative) opinion is re-sampled up to
        # this many extra times before we give up — see elicit_belief.
        self.max_flat_retries = max_flat_retries
        self.rate_limiter = rate_limiter
        self.parse_failures = 0  # replies that stayed flat after all re-samples
        self.transcript: list[dict] = []

        if backend == "anthropic":
            self.temperature = float(np.clip(temperature, 0.0, 1.0))
            if client is None:
                from anthropic import Anthropic
                client = Anthropic()
            self.client = client
        elif backend in ("groq", "openai", "vertex"):
            self.temperature = float(np.clip(temperature, 0.0, 2.0))
            if backend == "groq":
                self.base_url = base_url or GROQ_BASE_URL
                self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
            elif backend == "vertex":
                self.base_url = base_url or os.environ.get("VERTEX_BASE_URL", "")
                if not self.base_url:
                    raise ValueError("vertex backend needs base_url (…/endpoints/openapi)")
                self.api_key = ""  # token minted per-request from ADC via get_vertex_token()
            else:
                self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
                self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
            if client is None:
                import httpx
                client = httpx.Client(timeout=timeout)
            self.client = client
        else:
            raise ValueError(f"unknown backend: {backend!r} (expected groq | openai | anthropic)")

    def _build_prompt(self, question: str, answer_space: list[str],
                      neighbor_summary: str | None, evidence: str | None = None) -> str:
        # Options are LABELLED (A, B, C, …) and the model is asked to score by
        # label. Scoring by option *text* is what silently zeroed out AQuA: the
        # options there are long math expressions the model never reproduces
        # verbatim as JSON keys, so every score.get(text) missed and the opinion
        # collapsed to the flat zero state. Labels are short and echoed exactly.
        labels = [_LABELS[i] for i in range(len(answer_space))]
        options = "\n".join(f"{lab}. {a}" for lab, a in zip(labels, answer_space))
        # `evidence` is a NEUTRAL reference/context block (exp6C shared-information
        # mechanism): it injects a piece of evidence/rationale, framed as background
        # material — deliberately NOT as "other agents chose X" (that social framing
        # lives in neighbor_summary). This keeps exp6C's causal mechanism = common
        # evidence, distinct from the vote-copying of the coupling sweep.
        ev = (f"\n\nReference information or evidence for this question:\n{evidence}"
              if evidence else "")
        context = (f"\n\nCurrent views of the other agents in the group:\n{neighbor_summary}"
                   if neighbor_summary else "")
        role = f"{self.persona}\n\n" if self.persona else ""
        return (
            f"{role}Question: {question}\n\nOptions:\n{options}{ev}{context}\n\n"
            "Think briefly, then rate how strongly you believe EACH label "
            f"({', '.join(labels)}) is correct, on a scale from -10 to 10, where 10 means "
            "certainly correct. Reply with JSON ONLY, in the form "
            '{"scores": {"A": <number>, "B": <number>, ...}, "reasoning": "<brief>"}.'
        )

    def _parse_scores(self, text: str, answer_space: list[str]) -> dict[str, float]:
        # Free-form reasoning often breaks the outer JSON (raw newlines inside
        # strings, truncation at max_tokens), so fall back to parsing just the
        # "scores" sub-object, which comes first and survives truncation.
        candidates = []
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            candidates.append(match.group(0))
        sub = re.search(r'"scores"\s*:\s*(\{[^{}]*\})', text, re.DOTALL)
        if sub:
            candidates.append('{"scores": %s}' % sub.group(1))
        for cand in candidates:
            try:
                obj = json.loads(cand, strict=False)
            except Exception:
                continue
            scores = obj.get("scores", obj) if isinstance(obj, dict) else {}
            if isinstance(scores, dict) and scores:
                return self._resolve_scores(scores, answer_space)
        return {a: 0.0 for a in answer_space}

    def _resolve_scores(self, raw: dict, answer_space: list[str]) -> dict[str, float]:
        """Map the model's score keys onto answer positions, tolerant of the model
        keying by label (A/B/…), by option text, or by index. Options the model
        omitted are filled just below the lowest score it *did* give, so a reply
        that endorses only one option still yields a clear peak rather than a flat
        state. Returns a complete {answer: score} dict (flat zeros if nothing
        resolved, which triggers a re-sample upstream)."""
        n = len(answer_space)
        label_idx = {_LABELS[i].lower(): i for i in range(n)}
        text_idx = {_norm(a): i for i, a in enumerate(answer_space)}
        resolved: dict[int, float] = {}
        for k, v in raw.items():
            try:
                val = float(v)
            except (TypeError, ValueError):
                continue
            idx = _resolve_key(str(k), n, label_idx, text_idx, answer_space)
            if idx is not None:
                resolved.setdefault(idx, val)
        if not resolved:
            return {a: 0.0 for a in answer_space}
        fill = min(resolved.values()) - 1.0
        return {answer_space[i]: resolved.get(i, fill) for i in range(n)}

    def _retry_wait(self, err: Exception, attempt: int) -> float:
        """429 -> honor Retry-After (or long backoff); everything else -> short backoff."""
        resp = getattr(err, "response", None)
        if resp is not None and getattr(resp, "status_code", None) == 429:
            retry_after = resp.headers.get("retry-after") if hasattr(resp, "headers") else None
            if retry_after:
                try:
                    return float(retry_after) + 0.5
                except ValueError:
                    pass
            return min(60.0, 5.0 * 2.0 ** attempt)
        return 2.0 ** attempt

    def _chat(self, prompt: str) -> str:
        if self.rate_limiter is not None:
            self.rate_limiter.acquire()
        if self.backend == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(b.text for b in response.content if b.type == "text")

        if self.backend == "vertex":
            headers = {"Authorization": f"Bearer {get_vertex_token()}"}
        else:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        resp = self.client.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json={
                "model": self.model,
                "max_tokens": 512,
                "temperature": self.temperature,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _chat_with_retries(self, prompt: str) -> str:
        """One reply, retrying transient/429 errors with backoff (raises if all fail)."""
        for attempt in range(self.max_retries):
            try:
                return self._chat(prompt)
            except Exception as err:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(self._retry_wait(err, attempt))

    def elicit_belief(self, question: str, answer_space: list[str],
                       neighbor_summary: str | None = None,
                       evidence: str | None = None) -> np.ndarray:
        prompt = self._build_prompt(question, answer_space, neighbor_summary, evidence)
        # Re-sample while the reply parses to a flat (uninformative) opinion: a
        # flat state is almost always a parse/formatting failure, not a genuine
        # "all options equally likely" belief, and it would otherwise pollute the
        # dynamics and — with truth pinned at canonical index 0 — get scored as a
        # spurious correct vote. temperature > 0 makes a re-sample worth trying.
        x = np.zeros(len(answer_space))
        for _ in range(self.max_flat_retries + 1):
            text = self._chat_with_retries(prompt)
            scores = self._parse_scores(text, answer_space)
            self.transcript.append({"question": question, "response": text, "scores": scores})
            x = extract_opinion_state(scores, answer_space)
            if x.max() > x.min():  # informative → done
                return x
        self.parse_failures += 1
        return x
