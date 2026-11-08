"""SGIR confirmatory margin-gated intervention test. See SGIR_PREREG.md -- this script
implements it exactly; do not change the gate, population, split, or bootstrap here without
appending a dated amendment to that document first.

Two modes, strictly separated so a dev fit can be inspected before TEST is ever read:

  python sgir_confirmatory.py --fit-dev
      Fits tau_hat on the DEV split only (frozen TAU_GRID, stratum-matched random-budget
      comparator). Writes results/tables/sgir_dev_fit.json. Never reads TEST outcomes.

  python sgir_confirmatory.py --evaluate-test --i-have-frozen-the-protocol
      Single, one-time confirmatory evaluation on TEST using the tau_hat already written by
      --fit-dev (refuses to run if sgir_dev_fit.json is missing, or if the split manifest's
      sha256 does not match what --fit-dev was run against). Requires the explicit
      --i-have-frozen-the-protocol flag -- this is deliberately not the default behavior.
      Writes results/tables/sgir_confirmatory_test.json. Per SGIR_PREREG.md Sec 8, this is a
      SINGLE confirmatory run: re-running it after inspecting the output and changing anything
      upstream (tau selection objective, population, split, bootstrap spec) voids the
      pre-registration.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from sgir_common import (
    N_AGENTS, TABLES, apply_gate, assert_balanced_clusters, build_episodes,
    cluster_bootstrap_resample, cluster_groups, decompose_break_rescue, eligible_mask,
    manifest_hash, policy_outcome, random_budget_closed_form, random_budget_monte_carlo,
    select_threshold, stratum_k, stratum_labels)

N_BOOT = 20000
N_RANDOM_MC = 10000
BOOT_SEED = 20261108


def _load_manifest() -> dict:
    f = TABLES / "sgir_split_manifest.json"
    if not f.exists():
        raise SystemExit("sgir_split_manifest.json missing -- run sgir_split_manifest.py first "
                          "(SGIR_PREREG.md Sec 3)")
    manifest = json.loads(f.read_text())
    recomputed = manifest_hash(manifest["assignment"])
    if recomputed != manifest["sha256"]:
        raise SystemExit("split manifest hash mismatch -- the manifest was edited after being "
                          "frozen; this invalidates the protocol (SGIR_PREREG.md Sec 3)")
    return manifest


def _episode_arrays(episodes: list[dict]) -> dict:
    model = np.array([e["model"] for e in episodes])
    bench = np.array([e["bench"] for e in episodes])
    cluster = np.array([e["cluster"] for e in episodes])
    assert_balanced_clusters(cluster)   # SGIR_PREREG.md Sec 2.1 aggregation-equivalence guard
    return dict(
        margin=np.array([e["margin"] if e["margin"] is not None else np.nan for e in episodes],
                        float),
        keep_correct=np.array([e["keep_correct"] for e in episodes], bool),
        indep_correct=np.array([e["indep_correct"] for e in episodes], bool),
        keep_label=np.array([e["keep_label"] for e in episodes], int),
        indep_label=np.array([e["indep_label"] for e in episodes], int),
        cluster=cluster, model=model, bench=bench, stratum=stratum_labels(model, bench))


def _split_episodes(episodes: list[dict], manifest: dict) -> tuple[list[dict], list[dict]]:
    assignment = manifest["assignment"]
    dev, test = [], []
    for e in episodes:
        key = f"{e['model']}|{e['bench']}|{e['regime']}|{e['qi']}"
        split = assignment.get(key)
        if split == "dev":
            dev.append(e)
        elif split == "test":
            test.append(e)
        else:
            raise SystemExit(f"episode {key} not present in frozen split manifest")
    return dev, test


def cmd_fit_dev(_args: argparse.Namespace) -> None:
    manifest = _load_manifest()
    episodes = build_episodes()
    dev, _test = _split_episodes(episodes, manifest)
    arr = _episode_arrays(dev)
    fit = select_threshold(arr["margin"], arr["keep_correct"], arr["indep_correct"],
                           arr["stratum"])
    out = {"manifest_sha256": manifest["sha256"], "n_dev_episodes": len(dev), "fit": fit}
    dst = TABLES / "sgir_dev_fit.json"
    dst.write_text(json.dumps(out, indent=2))
    print(f"wrote {dst}")
    print(f"  n_dev_episodes={len(dev)}  tau_hat={fit['tau_hat']}  k_hat={fit['k_hat']}  "
          f"obj_hat={fit['obj_hat']:+.4f}")
    print(f"  k_by_stratum_hat={fit['k_by_stratum_hat']}")


def _nested_bootstrap(arr: dict, mask_fns: dict, gate_mask_for_random: np.ndarray, n_boot: int,
                       rng: np.random.Generator) -> dict:
    """mask_fns: name -> callable(arr, idx)->bool intervene mask (deterministic policies).
    Returns per-policy accuracy draws (n_boot,) plus RANDOM@budget draws, over cluster-
    resampled copies of arr, with one fresh per-stratum-matched random draw per resample for
    RANDOM@budget (SGIR_PREREG.md Sec 5 + the stratification amendment)."""
    uniq, groups = cluster_groups(arr["cluster"])
    draws = {name: np.empty(n_boot, float) for name in mask_fns}
    draws["random_budget"] = np.empty(n_boot, float)
    for b in range(n_boot):
        idx = cluster_bootstrap_resample(uniq, groups, rng)
        kc, ic = arr["keep_correct"][idx], arr["indep_correct"][idx]
        elig = eligible_mask(arr["margin"][idx])
        strata_b = arr["stratum"][idx]
        gate_mask_b = gate_mask_for_random[idx]
        for name, fn in mask_fns.items():
            mask = fn(arr, idx)
            draws[name][b] = float(policy_outcome(kc, ic, mask).mean())
        mc = random_budget_monte_carlo(kc, ic, elig, strata_b, gate_mask_b, 1, rng)
        draws["random_budget"][b] = mc[0]
    return draws


def _ci(x: np.ndarray) -> list[float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return [None, None]
    return [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))]


def cmd_evaluate_test(args: argparse.Namespace) -> None:
    if not args.i_have_frozen_the_protocol:
        raise SystemExit(
            "refusing to run --evaluate-test without --i-have-frozen-the-protocol. "
            "SGIR_PREREG.md, the split manifest, and tests/test_sgir_confirmatory.py must be "
            "frozen and signed off first. This is a single, one-time confirmatory evaluation "
            "(SGIR_PREREG.md Sec 8); re-running it after changing anything upstream voids it.")
    manifest = _load_manifest()
    dev_fit_path = TABLES / "sgir_dev_fit.json"
    if not dev_fit_path.exists():
        raise SystemExit("sgir_dev_fit.json missing -- run --fit-dev first")
    dev_fit = json.loads(dev_fit_path.read_text())
    if dev_fit["manifest_sha256"] != manifest["sha256"]:
        raise SystemExit("dev fit was computed against a different split manifest -- refit")
    tau_hat = dev_fit["fit"]["tau_hat"]
    if tau_hat is None:
        raise SystemExit("dev fit produced no eligible threshold -- cannot evaluate")

    episodes = build_episodes()
    _dev, test = _split_episodes(episodes, manifest)
    arr = _episode_arrays(test)
    elig = eligible_mask(arr["margin"])
    gate_mask = apply_gate(arr["margin"], tau_hat)
    k_test = int(gate_mask.sum())
    k_by_stratum_test = stratum_k(gate_mask, arr["stratum"])

    def keep_mask(a, idx):
        return np.zeros(len(idx), bool)

    def rerunall_mask(a, idx):
        return np.ones(len(idx), bool)

    def gate_mask_fn(a, idx):
        return apply_gate(a["margin"][idx], tau_hat)

    rng = np.random.default_rng(BOOT_SEED)
    draws = _nested_bootstrap(
        arr, {"keep": keep_mask, "rerunall": rerunall_mask, "gate": gate_mask_fn},
        gate_mask_for_random=gate_mask, n_boot=N_BOOT, rng=rng)

    acc_keep = float(policy_outcome(arr["keep_correct"], arr["indep_correct"],
                                    np.zeros(len(test), bool)).mean())
    acc_rerunall = float(policy_outcome(arr["keep_correct"], arr["indep_correct"],
                                        np.ones(len(test), bool)).mean())
    acc_gate = float(policy_outcome(arr["keep_correct"], arr["indep_correct"], gate_mask).mean())
    acc_random_cf = random_budget_closed_form(arr["keep_correct"], arr["indep_correct"],
                                              elig, arr["stratum"], gate_mask)
    mc_rng = np.random.default_rng([BOOT_SEED, 1])
    mc_draws = random_budget_monte_carlo(arr["keep_correct"], arr["indep_correct"], elig,
                                         arr["stratum"], gate_mask, N_RANDOM_MC, mc_rng)

    delta_gate_random = acc_gate - float(draws["random_budget"].mean())
    delta_gate_random_draws = draws["gate"] - draws["random_budget"]

    decomp = decompose_break_rescue(arr["keep_label"], arr["indep_label"], arr["keep_correct"],
                                     arr["indep_correct"], gate_mask)
    assert abs(decomp["delta_acc"] - decomp["delta_acc_check"]) < 1e-9, (
        "Prop 6.1 identity failed on the gated-in subset -- stop, this indicates a bug, not "
        "a result")

    out = {
        "manifest_sha256": manifest["sha256"],
        "tau_hat": tau_hat,
        "n_test_episodes": len(test),
        "n_eligible_test": int(elig.sum()),
        "k_test": k_test,
        "k_by_stratum_test": k_by_stratum_test,
        "intervention_fraction": k_test / int(elig.sum()) if elig.sum() else None,
        "extra_calls_per_question": (k_test / int(elig.sum()) * N_AGENTS) if elig.sum() else None,
        "accuracy": {
            "keep": {"point": acc_keep, "ci": _ci(draws["keep"])},
            "rerunall": {"point": acc_rerunall, "ci": _ci(draws["rerunall"])},
            "gate": {"point": acc_gate, "ci": _ci(draws["gate"])},
            "random_budget": {"point_closed_form": acc_random_cf,
                              "point_monte_carlo_mean": float(mc_draws.mean()),
                              "monte_carlo_ci": _ci(mc_draws),
                              "nested_bootstrap_mean": float(draws["random_budget"].mean()),
                              "nested_bootstrap_ci": _ci(draws["random_budget"])},
        },
        "estimands": {
            "delta_gate_random": {"point": delta_gate_random, "ci": _ci(delta_gate_random_draws)},
            "delta_gate_keep": {"point": acc_gate - acc_keep,
                                "ci": _ci(draws["gate"] - draws["keep"])},
            "delta_gate_rerunall": {"point": acc_gate - acc_rerunall,
                                    "ci": _ci(draws["gate"] - draws["rerunall"])},
        },
        "break_rescue_decomposition_gated_subset": decomp,
        "practical_effect_threshold_pp": 5.0,
        "note": "practical_effect_threshold_pp is an equivalence/magnitude label (Q2), never a "
                "significance criterion for delta_gate_random's CI (Q1). See SGIR_PREREG.md Sec 7. "
                "random_budget is matched to the gate's intervention count WITHIN EACH "
                "(model,bench) stratum, not a single pooled count (see k_by_stratum_test).",
    }
    dst = TABLES / "sgir_confirmatory_test.json"
    dst.write_text(json.dumps(out, indent=2))
    print(f"wrote {dst}")
    print(f"  Delta_gate-random = {delta_gate_random:+.4f}  CI={_ci(delta_gate_random_draws)}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fit-dev", action="store_true")
    g.add_argument("--evaluate-test", action="store_true")
    ap.add_argument("--i-have-frozen-the-protocol", action="store_true")
    args = ap.parse_args(argv)
    if args.fit_dev:
        cmd_fit_dev(args)
    else:
        cmd_evaluate_test(args)


if __name__ == "__main__":
    main()
