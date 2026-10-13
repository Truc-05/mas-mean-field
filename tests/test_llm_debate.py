import time

import numpy as np
import pytest

from mfg_llm.llm_debate.agent import DebateAgent, RateLimiter, get_rate_limiter


class _StubResponse:
    def __init__(self, content: str):
        self._content = content

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


class _StubClient:
    def __init__(self, content: str):
        self.content = content
        self.calls = []

    def post(self, url, headers=None, json=None):
        self.calls.append({"url": url, "json": json})
        return _StubResponse(self.content)


def test_groq_backend_elicit_belief_parses_scores():
    stub = _StubClient('{"scores": {"A": 5, "B": -2, "C": 0}, "reasoning": "ok"}')
    agent = DebateAgent(0, backend="groq", api_key="test", client=stub)
    x = agent.elicit_belief("q?", ["A", "B", "C"])
    assert x.shape == (3,)
    assert int(np.argmax(x)) == 0
    assert abs(float(np.mean(x))) < 1e-9
    assert stub.calls[0]["url"].endswith("/chat/completions")
    assert stub.calls[0]["json"]["model"] == agent.model


def test_unparseable_response_falls_back_to_neutral():
    stub = _StubClient("no json here")
    agent = DebateAgent(0, backend="groq", api_key="test", client=stub)
    x = agent.elicit_belief("q?", ["A", "B"])
    assert np.allclose(x, 0.0)


def test_retry_then_success():
    class FlakyClient(_StubClient):
        def __init__(self, content):
            super().__init__(content)
            self.attempts = 0

        def post(self, url, headers=None, json=None):
            self.attempts += 1
            if self.attempts == 1:
                raise ConnectionError("transient")
            return super().post(url, headers=headers, json=json)

    stub = FlakyClient('{"scores": {"A": 1, "B": 0}}')
    agent = DebateAgent(0, backend="groq", api_key="test", client=stub, max_retries=3)
    x = agent.elicit_belief("q?", ["A", "B"])
    assert stub.attempts == 2
    assert int(np.argmax(x)) == 0


def test_label_scoring_resolves_long_option_text():
    # the AQuA failure mode: options are long expressions the model never echoes
    # verbatim, but it scores by label, which must resolve to the right position.
    opts = ["5(√3 + 1)", "6(√3 + √2)", "7(√3 – 1)", "None of these"]
    stub = _StubClient('{"scores": {"A": 8, "B": -3, "C": 0, "D": -5}}')
    agent = DebateAgent(0, backend="groq", api_key="k", client=stub)
    x = agent.elicit_belief("q?", opts)
    assert int(np.argmax(x)) == 0  # label A -> canonical index 0
    assert x.max() > x.min()       # informative, not flat


def test_text_keyed_scores_still_resolve():
    stub = _StubClient('{"scores": {"Paris": 9, "London": 1, "Rome": -4}}')
    agent = DebateAgent(0, backend="groq", api_key="k", client=stub)
    x = agent.elicit_belief("capital?", ["Paris", "London", "Rome"])
    assert int(np.argmax(x)) == 0


def test_omitted_options_filled_below_min_give_a_peak():
    # model endorses only one option; the rest must land below it, not tie flat.
    stub = _StubClient('{"scores": {"A": 7}}')
    agent = DebateAgent(0, backend="groq", api_key="k", client=stub)
    x = agent.elicit_belief("q?", ["w", "x", "y", "z"])
    assert int(np.argmax(x)) == 0
    assert x.max() > x.min()


def test_flat_reply_is_resampled_then_used():
    class SequenceClient:
        def __init__(self, contents):
            self.contents = list(contents)
            self.calls = 0

        def post(self, url, headers=None, json=None):
            i = min(self.calls, len(self.contents) - 1)
            self.calls += 1
            return _StubResponse(self.contents[i])

    stub = SequenceClient(["garbage no json", '{"scores": {"A": 4, "B": 0}}'])
    agent = DebateAgent(0, backend="groq", api_key="k", client=stub, max_flat_retries=2)
    x = agent.elicit_belief("q?", ["A", "B"])
    assert stub.calls == 2              # first flat reply re-sampled once
    assert int(np.argmax(x)) == 0
    assert agent.parse_failures == 0


def test_persistent_flat_counts_as_parse_failure():
    stub = _StubClient("never any json")
    agent = DebateAgent(0, backend="groq", api_key="k", client=stub, max_flat_retries=2)
    x = agent.elicit_belief("q?", ["A", "B", "C"])
    assert np.allclose(x, 0.0)
    assert agent.parse_failures == 1   # gave up after 1 + max_flat_retries samples


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        DebateAgent(0, backend="mystery")


def test_temperature_clamped_per_backend():
    stub = _StubClient("{}")
    a = DebateAgent(0, backend="groq", api_key="k", client=stub, temperature=5.0)
    assert a.temperature == 2.0


def test_rate_limiter_spaces_requests():
    limiter = RateLimiter(rpm=6000)  # 10ms spacing
    t0 = time.monotonic()
    for _ in range(3):
        limiter.acquire()
    assert time.monotonic() - t0 >= 0.015


def test_get_rate_limiter_is_shared():
    assert get_rate_limiter(17.0) is get_rate_limiter(17.0)


def test_429_retry_honors_retry_after():
    class _Resp429:
        status_code = 429
        headers = {"retry-after": "0.01"}

    class _Err429(Exception):
        def __init__(self):
            self.response = _Resp429()

    class RateLimitedClient(_StubClient):
        def __init__(self, content):
            super().__init__(content)
            self.attempts = 0

        def post(self, url, headers=None, json=None):
            self.attempts += 1
            if self.attempts <= 2:
                raise _Err429()
            return super().post(url, headers=headers, json=json)

    stub = RateLimitedClient('{"scores": {"A": 3, "B": 0}}')
    agent = DebateAgent(0, backend="groq", api_key="k", client=stub, max_retries=5)
    t0 = time.monotonic()
    x = agent.elicit_belief("q?", ["A", "B"])
    elapsed = time.monotonic() - t0
    assert stub.attempts == 3
    assert int(np.argmax(x)) == 0
    assert elapsed < 2.0  # honored 0.01s retry-after, not the 5s+ default backoff
