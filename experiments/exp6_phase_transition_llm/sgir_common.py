"""Shared data layer + pure functions for the SGIR confirmatory margin-gated intervention
test. See SGIR_PREREG.md for the frozen protocol this module implements. Offline only -- no
LLM calls; reads the already-collected frozen Tier-3 raw dumps.

Every function below that participates in the gate decision or the dev/test split is written
to take plain arrays/dicts so it can be exercised by tests/test_sgir_confirmatory.py on
synthetic fixtures, with no dependency on the real data files.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from analysis_C_matched_D_falsification import margin, plur  # noqa: E402  frozen, verbatim
from intervention_validity import is_valid as _intervention_valid  # noqa: E402
from tier3_matched_d import BENCHMARKS, MODELS  # noqa: E402  frozen Tier-3 grid (3x3)

ROOT = HERE.parents[1]
PROC = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"

REGIMES = ["easy", "hard"]           # pooled as plain data partitions; NOT a scientific label
N_AGENTS = 11
R_REPS = 8
KEEP_COND = "shared_correct_strong"   # already-paid, dependence-inducing collective process
INDEP_COND = "indep_correct_strong"   # independence-restoring, matched-evidence re-reason
BASE_SPLIT_SEED = 20261108            # frozen date this protocol was written, not Date.now()

assert all(_intervention_valid(b) for b in BENCHMARKS), (
    "SGIR_PREREG.md population lock violated: a Tier-3 benchmark is invalid for the evidence "
    "intervention per intervention_validity.py -- population drifted, stop and amend the prereg")


def in_population(model: str, bench: str) -> bool:
    """Pure population filter. MODELS/BENCHMARKS are the frozen Tier-3 grid; anything else
    (mistral/phi4/deepseek, bbh/mmlu/arc/commonsenseqa) is out of scope for this test."""
    return model in MODELS and bench in BENCHMARKS and _intervention_valid(bench)


def cluster_id(model: str, bench: str, regime: str, qi: int) -> str:
    return f"{model}|{bench}|{regime}|{qi}"


# ---- real-data loading (touches files; everything below this point is pure) --------------
def _raw_path(model: str, bench: str, regime: str) -> Path:
    return PROC / f"exp6c_raw_{regime}_scaleup_{model}_{bench}_N11.json"


def load_raw_cell(model: str, bench: str, regime: str) -> dict | None:
    f = _raw_path(model, bench, regime)
    if not f.exists():
        return None
    return json.loads(f.read_text())["raw"]


def cell_question_count(model: str, bench: str, regime: str) -> int:
    """Structural count only (len of the question list) -- does not read margin, labels, or
    correctness. Safe to call before the protocol is frozen (used by the split-manifest
    builder, which must not touch outcome data)."""
    raw = load_raw_cell(model, bench, regime)
    if raw is None:
        return 0
    return len(raw[KEEP_COND])


def canonical_question_lists() -> dict[tuple[str, str], list[tuple[str, int]]]:
    """{(model,bench): [(regime, qi), ...]} in the fixed canonical order
    [(regime, qi) for regime in REGIMES for qi in range(n_q)]. Structural only (question
    identity + count), never reads margin/labels/correctness."""
    out: dict[tuple[str, str], list[tuple[str, int]]] = {}
    for model in MODELS:
        for bench in BENCHMARKS:
            items: list[tuple[str, int]] = []
            for regime in REGIMES:
                n_q = cell_question_count(model, bench, regime)
                items.extend((regime, qi) for qi in range(n_q))
            out[(model, bench)] = items
    return out


def build_episodes() -> list[dict]:
    """Real-data episode list: one dict per (model,bench,regime,qi,r). Reads margin/labels/
    correctness -- this is the function that is NOT called until a dev-fit or test-evaluation
    is explicitly requested."""
    episodes: list[dict] = []
    for model in MODELS:
        for bench in BENCHMARKS:
            for regime in REGIMES:
                raw = load_raw_cell(model, bench, regime)
                if raw is None:
                    continue
                keep_cond, indep_cond = raw[KEEP_COND], raw[INDEP_COND]
                for qi in range(len(keep_cond)):
                    cid = cluster_id(model, bench, regime, qi)
                    for r in range(len(keep_cond[qi])):
                        ks, is_ = keep_cond[qi][r], indep_cond[qi][r]
                        kl, il = plur(ks), plur(is_)
                        episodes.append(dict(
                            cluster=cid, model=model, bench=bench, regime=regime, qi=qi, r=r,
                            margin=margin(ks), keep_label=kl, indep_label=il,
                            keep_correct=(kl == 0), indep_correct=(il == 0)))
    return episodes


# ---- split construction (pure, deterministic; no outcome data) ---------------------------
def _stable_seed_component(s: str) -> int:
    """Deterministic across processes/machines -- NOT Python's builtin hash(), which is
    salted per-process (PYTHONHASHSEED) and would silently break split reproducibility."""
    digest = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def build_split(strata: dict[tuple[str, str], list[tuple[str, int]]],
                 base_seed: int = BASE_SPLIT_SEED) -> dict[tuple[str, str, str, int], str]:
    """Deterministic 50/50 split per (model,bench) stratum. Pure function of (strata, seed) --
    touches only question identity, never margin/correctness. Same inputs -> byte-identical
    output, always (required for the freeze/hash guarantee in SGIR_PREREG.md Sec 3)."""
    assignment: dict[tuple[str, str, str, int], str] = {}
    for (model, bench), items in strata.items():
        seed_seq = np.random.SeedSequence(
            [base_seed, _stable_seed_component(model), _stable_seed_component(bench)])
        rng = np.random.default_rng(seed_seq)
        order = rng.permutation(len(items))
        half = len(items) // 2
        for rank, idx in enumerate(order):
            regime, qi = items[idx]
            assignment[(model, bench, regime, qi)] = "dev" if rank < half else "test"
    return assignment


def manifest_hash(assignment_str_keyed: dict[str, str]) -> str:
    canon = json.dumps(assignment_str_keyed, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


# ---- gate / policies (pure; operate on arrays aligned to an episode list) -----------------
def eligible_mask(margins: np.ndarray) -> np.ndarray:
    return np.isfinite(margins)


def apply_gate(margins: np.ndarray, tau: float) -> np.ndarray:
    """Boolean INDEP-intervene mask. Depends ONLY on margin + eligibility -- never on
    keep_correct/indep_correct/keep_label/indep_label. (Exercised by the leakage-guard test.)"""
    elig = eligible_mask(margins)
    return elig & (margins <= tau)


def policy_outcome(keep_correct: np.ndarray, indep_correct: np.ndarray,
                    intervene_mask: np.ndarray) -> np.ndarray:
    return np.where(intervene_mask, indep_correct, keep_correct)


# ---- frozen candidate threshold grid (SGIR_PREREG.md Sec 4, point 3 of the 2026-11-08
# amendment): enumerated from the voting setup (N_AGENTS) ALONE, before any margin value is
# observed in dev or test. margin(state) = (c1-c2)/m for integers 0<=c2<=c1, c1+c2<=m,
# m = n_informative in [1, N_AGENTS] -- so every achievable margin is of the form d/m with
# 0<=d<=m<=N_AGENTS. This superset is fixed purely by N_AGENTS=11 and contains every value the
# real margin() function can ever return; it is NOT derived from observed data. -----------
def frozen_tau_grid(n_agents: int = N_AGENTS) -> list[float]:
    return sorted({d / m for m in range(1, n_agents + 1) for d in range(0, m + 1)})


TAU_GRID = frozen_tau_grid(N_AGENTS)


def stratum_labels(model: np.ndarray, bench: np.ndarray) -> np.ndarray:
    return np.array([f"{m}|{b}" for m, b in zip(model, bench)])


def random_budget_closed_form(keep_correct: np.ndarray, indep_correct: np.ndarray,
                               elig: np.ndarray, strata: np.ndarray,
                               gate_mask: np.ndarray) -> float:
    """E[Acc] of a random-budget comparator matched to the gate's OWN per-(model,bench)-stratum
    intervention count (SGIR_PREREG.md Sec 4/5, point 2 of the 2026-11-08 amendment): within
    each stratum, a uniformly random size-k_s INDEP subset of that stratum's eligible pool,
    k_s = (gate_mask & stratum==s).sum() -- never a single pooled k. This keeps RANDOM's
    model/benchmark composition identical to the gate's, so a composition shift cannot masquerade
    as a targeting effect. Exact by linearity of expectation -- no simulation needed here."""
    n_total = len(keep_correct)
    if n_total == 0:
        return float("nan")
    total_expected = 0.0
    for s in np.unique(strata):
        idx_s = np.where(strata == s)[0]
        elig_s = elig[idx_s]
        n_elig_s = int(elig_s.sum())
        kc_s, ic_s = keep_correct[idx_s], indep_correct[idx_s]
        base_s = float(kc_s[~elig_s].sum())
        if n_elig_s == 0:
            total_expected += base_s
            continue
        k_s = min(int(gate_mask[idx_s].sum()), n_elig_s)
        mean_keep_s = float(kc_s[elig_s].mean())
        mean_indep_s = float(ic_s[elig_s].mean())
        total_expected += base_s + k_s * mean_indep_s + (n_elig_s - k_s) * mean_keep_s
    return total_expected / n_total


def random_budget_monte_carlo(keep_correct: np.ndarray, indep_correct: np.ndarray,
                               elig: np.ndarray, strata: np.ndarray, gate_mask: np.ndarray,
                               n_draws: int, rng: np.random.Generator) -> np.ndarray:
    """Explicit resampling of the random assignment itself (not just its closed-form mean),
    per SGIR_PREREG.md Sec 5, with the SAME per-stratum budget matching as the closed form
    above: each draw picks an independent random size-k_s eligible subset for INDEP within
    EVERY (model,bench) stratum, never a single pooled-k draw across the whole test set."""
    n_total = len(keep_correct)
    strata_info = []
    for s in np.unique(strata):
        idx_s = np.where(strata == s)[0]
        elig_idx_s = idx_s[elig[idx_s]]
        ineligible_idx_s = idx_s[~elig[idx_s]]
        k_s = min(int(gate_mask[idx_s].sum()), len(elig_idx_s))
        strata_info.append((elig_idx_s, ineligible_idx_s, k_s))
    out = np.empty(n_draws, float)
    for i in range(n_draws):
        correct = 0.0
        for elig_idx_s, ineligible_idx_s, k_s in strata_info:
            correct += float(keep_correct[ineligible_idx_s].sum())
            if k_s <= 0:
                correct += float(keep_correct[elig_idx_s].sum())
                continue
            chosen = rng.choice(elig_idx_s, size=k_s, replace=False)
            mask_local = np.isin(elig_idx_s, chosen)
            correct += float(indep_correct[elig_idx_s[mask_local]].sum())
            correct += float(keep_correct[elig_idx_s[~mask_local]].sum())
        out[i] = correct / n_total if n_total else float("nan")
    return out


def stratum_k(mask: np.ndarray, strata: np.ndarray) -> dict:
    return {str(s): int(mask[strata == s].sum()) for s in np.unique(strata)}


def select_threshold(margins: np.ndarray, keep_correct: np.ndarray, indep_correct: np.ndarray,
                      strata: np.ndarray, tau_grid: list[float] | None = None) -> dict:
    """DEV-ONLY threshold fit. obj(tau) = Acc(pi_tau) - E[Acc(RANDOM@stratum-matched-budget)],
    closed form. candidates default to the FROZEN TAU_GRID (enumerated from N_AGENTS alone,
    SGIR_PREREG.md Sec 4 point 3) -- NOT derived from the observed margin values, so the search
    space cannot be adapted post-hoc to what dev happens to contain.
    Tie-break: smaller total k first, then smaller tau. Returns the full curve for auditability,
    including the per-stratum intervention counts at the selected tau."""
    grid = tau_grid if tau_grid is not None else TAU_GRID
    elig = eligible_mask(margins)
    curve = []
    best = None  # (sort_key, tau, k, obj)
    for tau in grid:
        mask = apply_gate(margins, tau)
        k = int(mask.sum())
        acc_pi = float(policy_outcome(keep_correct, indep_correct, mask).mean())
        acc_rand = random_budget_closed_form(keep_correct, indep_correct, elig, strata, mask)
        obj = acc_pi - acc_rand
        curve.append({"tau": float(tau), "k": k, "acc_pi": acc_pi, "acc_random": acc_rand,
                      "obj": obj})
        cand = (obj, -k, -float(tau))   # maximize obj; tie-break smaller k, then smaller tau
        if best is None or cand > best[0]:
            best = (cand, tau, k, obj)
    if best is None:
        return {"tau_hat": None, "k_hat": 0, "obj_hat": float("nan"), "curve": curve,
                "k_by_stratum_hat": {}}
    tau_hat = best[1]
    mask_hat = apply_gate(margins, tau_hat)
    return {"tau_hat": float(tau_hat), "k_hat": int(best[2]), "obj_hat": float(best[3]),
            "k_by_stratum_hat": stratum_k(mask_hat, strata), "curve": curve}


# ---- break/rescue/churn decomposition (pure; Theorem 6 partition) ------------------------
def decompose_break_rescue(keep_label: np.ndarray, indep_label: np.ndarray,
                            keep_correct: np.ndarray, indep_correct: np.ndarray,
                            mask: np.ndarray) -> dict:
    idx = np.where(mask)[0]
    n = len(idx)
    if n == 0:
        nan = float("nan")
        return dict(n=0, p_break=nan, p_rescue=nan, p_churn_same=nan, p_churn_diff=nan,
                    p_stay_correct=nan, partition_sum=nan, delta_acc=nan,
                    delta_acc_check=nan, dY=nan)
    kc, ic = keep_correct[idx], indep_correct[idx]
    kl, il = keep_label[idx], indep_label[idx]
    p_break = float(np.mean(kc & ~ic))
    p_rescue = float(np.mean(~kc & ic))
    both_wrong = (~kc) & (~ic)
    same_label = kl == il
    p_churn_same = float(np.mean(both_wrong & same_label))
    p_churn_diff = float(np.mean(both_wrong & ~same_label))
    p_stay_correct = float(np.mean(kc & ic))
    return dict(
        n=int(n), p_break=p_break, p_rescue=p_rescue, p_churn_same=p_churn_same,
        p_churn_diff=p_churn_diff, p_stay_correct=p_stay_correct,
        partition_sum=p_break + p_rescue + p_churn_same + p_churn_diff + p_stay_correct,
        delta_acc=p_rescue - p_break,
        delta_acc_check=float(ic.mean()) - float(kc.mean()),
        dY=float(np.mean(kl != il)))


# ---- cluster bootstrap (pure; generic over any per-episode statistic) ---------------------
def cluster_groups(cluster_of_episode: np.ndarray) -> tuple[np.ndarray, dict]:
    uniq = np.unique(cluster_of_episode)
    groups = {c: np.where(cluster_of_episode == c)[0] for c in uniq}
    return uniq, groups


def cluster_bootstrap_resample(uniq: np.ndarray, groups: dict,
                                rng: np.random.Generator) -> np.ndarray:
    """One resample's worth of EPISODE INDICES (with repeats), built by resampling whole
    clusters with replacement -- a cluster's episodes are always kept together."""
    chosen = rng.choice(uniq, size=len(uniq), replace=True)
    return np.concatenate([groups[c] for c in chosen]) if len(chosen) else np.array([], int)


# ---- realization/question aggregation guard (SGIR_PREREG.md Sec 2.1, point 1 of the
# 2026-11-08 amendment) ---------------------------------------------------------------------
def assert_balanced_clusters(cluster_of_episode: np.ndarray) -> int:
    """The gate acts PER REALIZATION (per (question, replicate) episode), not per question;
    the dev/test SPLIT acts per question (a whole cluster always lands in one split). Flat
    pooling of accuracy over episodes is numerically identical to first averaging within each
    question and then averaging across questions ONLY IF every question-cluster contributes
    the same number of episodes. This must hold (R=8 uniformly in the frozen Tier-3 grid);
    raises loudly if a subset (e.g. after splitting) ever breaks that balance instead of
    silently reporting a flat-pooled number that would then mean something different."""
    if len(cluster_of_episode) == 0:
        return 0
    _, groups = cluster_groups(cluster_of_episode)
    sizes = {len(v) for v in groups.values()}
    if len(sizes) != 1:
        raise ValueError(
            f"unbalanced question-clusters: sizes observed = {sorted(sizes)} -- the flat-"
            f"pooling == per-question-average equivalence asserted in SGIR_PREREG.md Sec 2.1 "
            f"no longer holds; fix before fitting or evaluating, do not pool anyway")
    return sizes.pop()
