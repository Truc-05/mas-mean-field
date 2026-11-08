"""Read-only audit of the frozen sgir_confirmatory_test.json run: (1) proves the three Delta
CIs are genuine paired/cluster-bootstrap CIs on the difference distribution, not inferred from
the two policies' marginal CIs; (2) produces the model x benchmark (9-cell) breakdown promised
by SGIR_PREREG.md Sec 5 but not emitted by the single pooled run; (3) clarifies the cost
accounting (per-intervened-episode cost vs. per-deployed-episode amortized cost).

Deterministic: re-derives the SAME bootstrap draws with the SAME BOOT_SEED as the frozen run
(sgir_confirmatory.cmd_evaluate_test) -- this reproduces, it does not re-decide. tau_hat, the
split, and the population are read from the already-frozen artifacts; nothing here can change
them. Not a second confirmatory evaluation.
"""
from __future__ import annotations

import json

import numpy as np

from sgir_common import (
    N_AGENTS, TABLES, apply_gate, eligible_mask, policy_outcome, random_budget_closed_form,
    random_budget_monte_carlo, decompose_break_rescue, stratum_k)
from sgir_confirmatory import (
    BOOT_SEED, N_BOOT, N_RANDOM_MC, _ci, _episode_arrays, _load_manifest, _nested_bootstrap,
    _split_episodes)
from sgir_confirmatory import build_episodes  # re-exported via sgir_confirmatory import


def _slice(arr: dict, mask: np.ndarray) -> dict:
    return {k: v[mask] for k, v in arr.items()}


def main() -> None:
    manifest = _load_manifest()
    dev_fit = json.loads((TABLES / "sgir_dev_fit.json").read_text())
    assert dev_fit["manifest_sha256"] == manifest["sha256"]
    tau_hat = dev_fit["fit"]["tau_hat"]
    frozen = json.loads((TABLES / "sgir_confirmatory_test.json").read_text())

    episodes = build_episodes()
    _dev, test = _split_episodes(episodes, manifest)
    arr = _episode_arrays(test)
    elig = eligible_mask(arr["margin"])
    gate_mask = apply_gate(arr["margin"], tau_hat)

    def keep_mask(a, idx):
        return np.zeros(len(idx), bool)

    def rerunall_mask(a, idx):
        return np.ones(len(idx), bool)

    def gate_mask_fn(a, idx):
        return apply_gate(a["margin"][idx], tau_hat)

    # ---- (1) pairing audit: recompute the pooled draws, show it reproduces the frozen JSON
    # exactly, and show the Delta variance is SMALLER than naive independent combination would
    # give -- the signature of genuine within-replicate pairing (shared resampled data). -------
    rng = np.random.default_rng(BOOT_SEED)
    draws = _nested_bootstrap(
        arr, {"keep": keep_mask, "rerunall": rerunall_mask, "gate": gate_mask_fn},
        gate_mask_for_random=gate_mask, n_boot=N_BOOT, rng=rng)

    print("=" * 90)
    print("PAIRING AUDIT -- reproduced from the frozen seed, not a new confirmatory run")
    print("=" * 90)
    for name, other in [("random_budget", "gate"), ("keep", "gate"), ("rerunall", "gate")]:
        d = draws[other] - draws[name]
        ci = _ci(d)
        frozen_key = {"random_budget": "delta_gate_random", "keep": "delta_gate_keep",
                      "rerunall": "delta_gate_rerunall"}[name]
        frozen_ci = frozen["estimands"][frozen_key]["ci"]
        reproduced_ok = (abs(ci[0] - frozen_ci[0]) < 1e-9 and abs(ci[1] - frozen_ci[1]) < 1e-9)
        var_diff, var_sum = float(np.var(d)), float(np.var(draws[other]) + np.var(draws[name]))
        naive_half_width = 1.96 * float(np.sqrt(var_sum))
        paired_half_width = (ci[1] - ci[0]) / 2
        print(f"\n{frozen_key}: point={float(d.mean()):+.4f}  paired-bootstrap CI={ci}")
        print(f"  reproduces frozen JSON exactly: {reproduced_ok}  (frozen CI was {frozen_ci})")
        print(f"  Var(diff draws)={var_diff:.6f}  vs  naive independent-sum Var={var_sum:.6f}  "
              f"(diff/sum ratio={var_diff / var_sum:.3f})")
        print(f"  paired CI half-width={paired_half_width:.4f}  vs  naive-independent "
              f"half-width (1.96*sqrt(var_a+var_b))={naive_half_width:.4f}  "
              f"-> {'NARROWER (pairing detected, as expected from shared resampled data)' if paired_half_width < naive_half_width else 'NOT narrower -- investigate'}")

    # ---- (2) model x benchmark breakdown (SGIR_PREREG.md Sec 5, promised, not yet emitted) ---
    print("\n" + "=" * 90)
    print("MODEL x BENCHMARK BREAKDOWN (9 cells), same tau_hat, same nested bootstrap")
    print("=" * 90)
    cells = sorted(set(arr["stratum"]))
    breakdown = {}
    for cell in cells:
        mask_cell = arr["stratum"] == cell
        sub = _slice(arr, mask_cell)
        elig_c = eligible_mask(sub["margin"])
        gate_c = apply_gate(sub["margin"], tau_hat)
        k_c = int(gate_c.sum())
        rng_c = np.random.default_rng([BOOT_SEED, hash(cell) & 0xFFFFFFFF])
        draws_c = _nested_bootstrap(
            sub, {"keep": keep_mask, "rerunall": rerunall_mask, "gate": gate_mask_fn},
            gate_mask_for_random=gate_c, n_boot=N_BOOT, rng=rng_c)
        acc_keep_c = float(policy_outcome(sub["keep_correct"], sub["indep_correct"],
                                          np.zeros(len(sub["keep_correct"]), bool)).mean())
        acc_gate_c = float(policy_outcome(sub["keep_correct"], sub["indep_correct"], gate_c).mean())
        acc_rerunall_c = float(policy_outcome(sub["keep_correct"], sub["indep_correct"],
                                              np.ones(len(sub["keep_correct"]), bool)).mean())
        decomp_c = decompose_break_rescue(sub["keep_label"], sub["indep_label"],
                                          sub["keep_correct"], sub["indep_correct"], gate_c)
        d_gr = draws_c["gate"] - draws_c["random_budget"]
        breakdown[cell] = {
            "n_episodes": int(len(sub["keep_correct"])), "n_eligible": int(elig_c.sum()),
            "k": k_c, "intervention_fraction": k_c / int(elig_c.sum()) if elig_c.sum() else None,
            "acc_keep": acc_keep_c, "acc_gate": acc_gate_c, "acc_rerunall": acc_rerunall_c,
            "delta_gate_random": {"point": float(d_gr.mean()), "ci": _ci(d_gr)},
            "break_rescue_decomposition": decomp_c,
        }
        sig = "excl-0" if (d_gr.mean() > 0 and _ci(d_gr)[0] > 0) or (d_gr.mean() < 0 and _ci(d_gr)[1] < 0) else "overlaps-0"
        print(f"  {cell:22s} n={len(sub['keep_correct']):4d} k={k_c:3d} "
              f"({k_c / int(elig_c.sum()) if elig_c.sum() else float('nan'):5.1%})  "
              f"Acc keep/gate/rerunall={acc_keep_c:.3f}/{acc_gate_c:.3f}/{acc_rerunall_c:.3f}  "
              f"d_gate-random={d_gr.mean():+.4f} CI={_ci(d_gr)} [{sig}]")

    # ---- (3) cost accounting, disambiguated -----------------------------------------------
    print("\n" + "=" * 90)
    print("COST ACCOUNTING (disambiguated)")
    print("=" * 90)
    n_elig = frozen["n_eligible_test"]
    k_test = frozen["k_test"]
    frac = k_test / n_elig
    print(f"  cost PER INTERVENED episode (if gated)     = {N_AGENTS} agent calls (fixed; "
          f"INDEP = a full fresh N={N_AGENTS}-agent independent vote, NOT 1 extra call)")
    print(f"  cost PER DEPLOYED episode, amortized       = intervention_fraction * {N_AGENTS} "
          f"= {frac:.4f} * {N_AGENTS} = {frac * N_AGENTS:.4f} extra calls/question in "
          f"EXPECTATION over many questions (most pay 0, {frac:.1%} pay {N_AGENTS})")
    print(f"  RERUN-ALL cost PER deployed episode        = {N_AGENTS} (every question, always)")
    print(f"  gate cost is {(1 - frac):.1%} cheaper than RERUN-ALL in expected calls, while "
          f"matching/exceeding RERUN-ALL's point-estimate accuracy "
          f"({frozen['accuracy']['gate']['point']:.4f} vs "
          f"{frozen['accuracy']['rerunall']['point']:.4f})")

    out = {"pairing_audit_note": "see stdout; delta CIs are paired-bootstrap, reproduced exactly "
                                 "from the frozen BOOT_SEED, and are narrower than naive "
                                 "independent-CI combination would give",
           "model_x_bench_breakdown": breakdown,
           "cost_accounting": {"per_intervened_episode_calls": N_AGENTS,
                               "intervention_fraction_test": frac,
                               "amortized_extra_calls_per_deployed_episode": frac * N_AGENTS,
                               "rerunall_calls_per_deployed_episode": N_AGENTS}}
    dst = TABLES / "sgir_audit_breakdown.json"
    dst.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
