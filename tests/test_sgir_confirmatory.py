"""CPU-only tests for the SGIR confirmatory margin-gated intervention test
(SGIR_PREREG.md). All fixtures are hand-constructed; none of this touches the real frozen
Tier-3 data files. Covers: leakage, population filtering, stratum-matched budget matching,
break/rescue decomposition, cluster bootstrap, split determinism, the frozen (data-
independent) tau grid, aggregation balance, and threshold selection (incl. tie-break).
"""
import sys
from pathlib import Path

import numpy as np

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments" / "exp6_phase_transition_llm"
sys.path.insert(0, str(EXPERIMENTS))

from sgir_common import (  # noqa: E402
    N_AGENTS, TAU_GRID, apply_gate, assert_balanced_clusters, build_split,
    cluster_bootstrap_resample, cluster_groups, decompose_break_rescue, frozen_tau_grid,
    in_population, manifest_hash, policy_outcome, random_budget_closed_form,
    random_budget_monte_carlo, select_threshold, stratum_k, stratum_labels)


# ---- population filter ---------------------------------------------------------------------
def test_population_filter_accepts_frozen_grid_only():
    assert in_population("qwen", "truthfulqa")
    assert in_population("llama", "strategyqa")
    assert in_population("gemma", "aqua")


def test_population_filter_rejects_dropped_models_and_invalid_benchmarks():
    assert not in_population("mistral", "truthfulqa")   # dropped at Tier-3 promotion
    assert not in_population("phi4", "aqua")
    assert not in_population("deepseek", "strategyqa")
    assert not in_population("qwen", "bbh")              # invalid for the evidence intervention
    assert not in_population("qwen", "mmlu")             # Tier-1 breadth, not Tier-3


# ---- frozen, data-independent threshold grid (amendment point 3) ---------------------------
def test_tau_grid_is_fully_enumerable_from_n_agents_alone():
    grid = frozen_tau_grid(11)
    expected = sorted({d / m for m in range(1, 12) for d in range(0, m + 1)})
    assert grid == expected
    assert grid[0] == 0.0 and grid[-1] == 1.0
    assert all(0.0 <= t <= 1.0 for t in grid)
    assert grid == sorted(grid)              # monotone, no accidental duplicates-as-different


def test_tau_grid_module_constant_matches_frozen_n_agents():
    assert TAU_GRID == frozen_tau_grid(N_AGENTS)


def test_tau_grid_is_deterministic_and_data_independent():
    # calling it twice, with no data in scope at all, gives byte-identical output
    assert frozen_tau_grid(11) == frozen_tau_grid(11)
    # it does not take margins/outcomes as input at all -- there is no code path by which
    # observed dev values could change it (enforced structurally: the function's only
    # parameter is n_agents).
    import inspect
    assert list(inspect.signature(frozen_tau_grid).parameters) == ["n_agents"]


# ---- leakage guard: the gate decision depends only on margin + eligibility -----------------
def test_gate_decision_depends_only_on_margin_not_on_outcomes():
    margins = np.array([0.1, 0.4, 0.4, 0.9, np.nan])
    tau = 0.4
    mask_a = apply_gate(margins, tau)
    mask_b = apply_gate(margins.copy(), tau)
    assert np.array_equal(mask_a, mask_b)
    assert list(mask_a) == [True, True, True, False, False]  # nan is ineligible -> never gated


def test_threshold_curve_k_depends_only_on_margin_not_on_correctness():
    margins = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    strata = np.array(["s0"] * 5)
    keep_a = np.array([True, True, False, False, True])
    indep_a = np.array([False, True, True, False, True])
    keep_b = ~keep_a          # completely different outcome realization, same margins
    indep_b = ~indep_a
    grid = list(margins)      # restrict to a small grid for a tight, readable test
    fit_a = select_threshold(margins, keep_a, indep_a, strata, tau_grid=grid)
    fit_b = select_threshold(margins, keep_b, indep_b, strata, tau_grid=grid)
    ks_a = [row["k"] for row in fit_a["curve"]]
    ks_b = [row["k"] for row in fit_b["curve"]]
    taus_a = [row["tau"] for row in fit_a["curve"]]
    taus_b = [row["tau"] for row in fit_b["curve"]]
    assert ks_a == ks_b                 # intervention COUNT per tau is outcome-independent
    assert taus_a == taus_b             # candidate grid is outcome-independent
    objs_a = [row["obj"] for row in fit_a["curve"]]
    objs_b = [row["obj"] for row in fit_b["curve"]]
    assert objs_a != objs_b             # the OBJECTIVE value is allowed to differ


# ---- stratum-matched budget (amendment point 2) ---------------------------------------------
def test_random_budget_is_matched_per_stratum_not_pooled():
    # Two strata with very different base rates. The gate intervenes on ALL of stratum "hi"
    # (where INDEP is useless) and NONE of stratum "lo" (where INDEP would help a lot). A
    # pooled-k random comparator would spread its k draws across both strata by chance and
    # partly land in "lo" (raising random's accuracy); a stratum-matched comparator must keep
    # every one of its interventions inside "hi", same as the gate, by construction.
    n_each = 20
    strata = np.array(["hi"] * n_each + ["lo"] * n_each)
    keep_correct = np.array([True] * n_each + [False] * n_each)     # hi already correct
    indep_correct = np.array([True] * n_each + [True] * n_each)     # lo recoverable via INDEP
    elig = np.ones(2 * n_each, bool)
    gate_mask = np.array([True] * n_each + [False] * n_each)        # gate = all of "hi" only

    acc_random_cf = random_budget_closed_form(keep_correct, indep_correct, elig, strata, gate_mask)
    # stratum-matched: k_hi=n_each (all of hi, no-op since hi is already correct either way),
    # k_lo=0 (lo stays KEEP, which is wrong) -> E[Acc] = n_each(hi correct)/40 = 0.5 exactly.
    assert abs(acc_random_cf - 0.5) < 1e-12

    # A naive POOLED-k comparator (k = n_each total, drawn from the full 40) would instead
    # expect some of its budget to land in "lo" and recover extra accuracy there -- i.e. a
    # strictly higher number than the stratified 0.5. Demonstrate the two are NOT the same
    # computation (guards against silently reverting to pooled matching).
    pooled_mean_keep = float(keep_correct.mean())
    pooled_mean_indep = float(indep_correct.mean())
    k_pooled = int(gate_mask.sum())
    naive_pooled = (k_pooled * pooled_mean_indep + (2 * n_each - k_pooled) * pooled_mean_keep) / (2 * n_each)
    assert naive_pooled > acc_random_cf + 0.05


def test_random_budget_monte_carlo_matches_closed_form_per_stratum():
    rng = np.random.default_rng(321)
    n_each = 30
    strata = np.array(["a"] * n_each + ["b"] * n_each)
    keep_correct = rng.random(2 * n_each) < 0.4
    indep_correct = rng.random(2 * n_each) < 0.7
    elig = np.ones(2 * n_each, bool)
    gate_mask = np.zeros(2 * n_each, bool)
    gate_mask[:10] = True       # k_a = 10, k_b = 0
    cf = random_budget_closed_form(keep_correct, indep_correct, elig, strata, gate_mask)
    mc = random_budget_monte_carlo(keep_correct, indep_correct, elig, strata, gate_mask, 8000,
                                   np.random.default_rng(322))
    assert abs(cf - mc.mean()) < 0.03


def test_random_budget_monte_carlo_intervenes_exactly_k_per_stratum():
    # within each stratum, KEEP always wrong, INDEP always right -> accuracy reveals exactly
    # how many of each stratum's k got INDEP, every single draw.
    strata = np.array(["a"] * 10 + ["b"] * 10)
    keep_correct = np.zeros(20, bool)
    indep_correct = np.ones(20, bool)
    elig = np.ones(20, bool)
    gate_mask = np.zeros(20, bool)
    gate_mask[:4] = True     # k_a = 4
    gate_mask[10:13] = True  # k_b = 3
    rng = np.random.default_rng(55)
    draws = random_budget_monte_carlo(keep_correct, indep_correct, elig, strata, gate_mask,
                                      500, rng)
    expected = (4 + 3) / 20
    assert np.allclose(draws, expected)


def test_stratum_k_reports_per_stratum_counts():
    mask = np.array([True, False, True, True, False])
    strata = np.array(["x", "x", "y", "y", "y"])
    assert stratum_k(mask, strata) == {"x": 1, "y": 2}


def test_closed_form_random_budget_boundary_cases():
    rng = np.random.default_rng(7)
    keep_correct = rng.random(50) < 0.4
    indep_correct = rng.random(50) < 0.7
    elig = np.ones(50, bool)
    elig[:5] = False
    strata = np.array(["only"] * 50)
    acc_keep_full = float(policy_outcome(keep_correct, indep_correct, np.zeros(50, bool)).mean())
    zero_mask = np.zeros(50, bool)
    assert abs(random_budget_closed_form(keep_correct, indep_correct, elig, strata, zero_mask)
               - acc_keep_full) < 1e-12
    acc_all_elig_indep = float(policy_outcome(keep_correct, indep_correct, elig).mean())
    assert abs(random_budget_closed_form(keep_correct, indep_correct, elig, strata, elig)
               - acc_all_elig_indep) < 1e-12


# ---- break/rescue/churn decomposition --------------------------------------------------------
def test_decomposition_partition_and_identity():
    keep_label = np.array([0, 1, 1, 1, 0])
    indep_label = np.array([1, 0, 1, 2, 0])
    keep_correct = np.array([True, False, False, False, True])
    indep_correct = np.array([False, True, False, False, True])
    mask = np.ones(5, bool)
    d = decompose_break_rescue(keep_label, indep_label, keep_correct, indep_correct, mask)
    assert d["n"] == 5
    assert abs(d["p_break"] - 0.2) < 1e-12
    assert abs(d["p_rescue"] - 0.2) < 1e-12
    assert abs(d["p_churn_same"] - 0.2) < 1e-12
    assert abs(d["p_churn_diff"] - 0.2) < 1e-12
    assert abs(d["p_stay_correct"] - 0.2) < 1e-12
    assert abs(d["partition_sum"] - 1.0) < 1e-12
    assert abs(d["delta_acc"] - d["delta_acc_check"]) < 1e-12
    assert abs(d["delta_acc"] - 0.0) < 1e-12
    assert abs(d["dY"] - 3 / 5) < 1e-12


def test_decomposition_empty_mask_returns_nan_not_zero():
    keep_label = np.array([0, 0])
    indep_label = np.array([0, 0])
    keep_correct = np.array([True, True])
    indep_correct = np.array([True, True])
    d = decompose_break_rescue(keep_label, indep_label, keep_correct, indep_correct,
                               np.zeros(2, bool))
    assert d["n"] == 0
    assert np.isnan(d["p_break"]) and np.isnan(d["delta_acc"])


# ---- cluster bootstrap ------------------------------------------------------------------------
def test_cluster_bootstrap_never_splits_a_cluster():
    cluster_of_episode = np.array(["c0", "c0", "c1", "c1", "c1", "c2"])
    uniq, groups = cluster_groups(cluster_of_episode)
    rng = np.random.default_rng(42)
    for _ in range(200):
        idx = cluster_bootstrap_resample(uniq, groups, rng)
        for c in uniq:
            member_set = set(groups[c].tolist())
            occurrences = [i for i in idx if i in member_set]
            assert len(occurrences) % len(member_set) == 0
            for start in range(0, len(occurrences), len(member_set)):
                block = occurrences[start:start + len(member_set)]
                assert set(block) == member_set


def test_cluster_bootstrap_deterministic_given_seed():
    cluster_of_episode = np.array(["a", "a", "b", "c", "c", "c"])
    uniq, groups = cluster_groups(cluster_of_episode)
    r1 = cluster_bootstrap_resample(uniq, groups, np.random.default_rng(5))
    r2 = cluster_bootstrap_resample(uniq, groups, np.random.default_rng(5))
    assert np.array_equal(r1, r2)


# ---- realization/question aggregation guard (amendment point 1) -----------------------------
def test_assert_balanced_clusters_passes_on_uniform_replicate_counts():
    cluster_of_episode = np.array(["q0", "q0", "q1", "q1"])  # 2 replicates each
    assert assert_balanced_clusters(cluster_of_episode) == 2


def test_assert_balanced_clusters_raises_on_uneven_replicate_counts():
    cluster_of_episode = np.array(["q0", "q0", "q1"])  # q1 has only 1 replicate
    try:
        assert_balanced_clusters(cluster_of_episode)
        assert False, "expected ValueError for unbalanced clusters"
    except ValueError:
        pass


def test_assert_balanced_clusters_empty_is_a_noop():
    assert assert_balanced_clusters(np.array([])) == 0


def test_stratum_labels_builds_model_bench_keys():
    labels = stratum_labels(np.array(["qwen", "llama"]), np.array(["aqua", "strategyqa"]))
    assert list(labels) == ["qwen|aqua", "llama|strategyqa"]


# ---- split determinism -----------------------------------------------------------------------
def test_split_is_deterministic_and_balanced():
    strata = {("m1", "b1"): [("easy", i) for i in range(10)] + [("hard", i) for i in range(10)]}
    a1 = build_split(strata, base_seed=20261108)
    a2 = build_split(strata, base_seed=20261108)
    assert a1 == a2
    n_dev = sum(1 for v in a1.values() if v == "dev")
    n_test = sum(1 for v in a1.values() if v == "test")
    assert n_dev == 10 and n_test == 10


def test_split_differs_by_stratum_seed_component():
    strata = {
        ("m1", "b1"): [("easy", i) for i in range(10)] + [("hard", i) for i in range(10)],
        ("m2", "b1"): [("easy", i) for i in range(10)] + [("hard", i) for i in range(10)],
    }
    assignment = build_split(strata, base_seed=20261108)
    a_m1 = {k: v for k, v in assignment.items() if k[0] == "m1"}
    a_m2 = {k: v for k, v in assignment.items() if k[0] == "m2"}
    dev_qis_m1 = sorted(k[3] for k, v in a_m1.items() if v == "dev" and k[2] == "easy")
    dev_qis_m2 = sorted(k[3] for k, v in a_m2.items() if v == "dev" and k[2] == "easy")
    assert dev_qis_m1 != dev_qis_m2


def test_manifest_hash_is_stable_and_order_independent():
    d1 = {"b": "test", "a": "dev"}
    d2 = {"a": "dev", "b": "test"}
    assert manifest_hash(d1) == manifest_hash(d2)
    d3 = {"a": "dev", "b": "dev"}
    assert manifest_hash(d1) != manifest_hash(d3)


# ---- threshold selection: a constructed case with a known-optimal interior tau -----------------
def test_select_threshold_finds_known_interior_optimum():
    margins = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    strata = np.array(["only"] * 10)
    keep_correct = margins > 0.5
    indep_correct = np.ones(10, bool)
    fit = select_threshold(margins, keep_correct, indep_correct, strata, tau_grid=list(margins))
    assert abs(fit["tau_hat"] - 0.5) < 1e-12
    assert fit["k_hat"] == 5
    assert abs(fit["obj_hat"] - 0.25) < 1e-9
    assert fit["k_by_stratum_hat"] == {"only": 5}


def test_select_threshold_tie_break_prefers_fewer_interventions():
    margins = np.array([0.1, 0.3, 0.7])
    strata = np.array(["only"] * 3)
    keep_correct = np.array([True, True, True])
    indep_correct = np.array([True, True, True])
    fit = select_threshold(margins, keep_correct, indep_correct, strata, tau_grid=list(margins))
    assert abs(fit["obj_hat"] - 0.0) < 1e-12
    assert fit["k_hat"] == 1
    assert abs(fit["tau_hat"] - 0.1) < 1e-12


def test_select_threshold_default_grid_is_the_frozen_tau_grid():
    # With no explicit tau_grid, select_threshold must use the module-level frozen TAU_GRID
    # (derived from N_AGENTS alone) -- never np.unique(observed margins).
    margins = np.array([0.37, 0.81])   # values that are NOT exactly on the N=11 grid
    strata = np.array(["only"] * 2)
    keep_correct = np.array([True, False])
    indep_correct = np.array([True, True])
    fit = select_threshold(margins, keep_correct, indep_correct, strata)
    observed_taus = {row["tau"] for row in fit["curve"]}
    assert observed_taus == set(TAU_GRID)
    assert 0.37 not in observed_taus and 0.81 not in observed_taus
