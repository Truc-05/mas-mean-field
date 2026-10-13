"""Exp6C BOUNDARY analysis — does the (large ΔD, small ΔAcc) puzzle resolve into a
SUSCEPTIBILITY-GATED effect? Offline; reads exp6c_boundary_{regime}_N{N}.json.

The claim under test (pre-registered direction):
    dependence created by sharing harms the crowd MAINLY near its decision boundary.
Operationally, with per-question
    ΔA_q = A_shared_q − A_indep_q            (matched-quality collective effect)
    ΔD_q = D_shared_q − D_indep_q            (dependence CREATED by sharing)
    M_q  = pre-intervention margin from the `none` votes (default: runner-up gap),
we look at three things:
  (1) ΔA vs M binned  — is |ΔA| concentrated at small M (contested questions)?
  (2) |ΔA| ~ M         — margin main effect (D-free, robust to noisy D_q).
  (3) ΔA ~ D + M + D·M — the interaction β3: does dependence bite harder at small M?
All CIs are question-level nonparametric bootstrap. D_q at R=8 is noisy, so (2) — which
does not use D — is the robust primary; (3) is corroborating. Raw states are persisted
(data/processed/exp6c_raw_*), so a shrinkage/mixed-model D_q can be swapped in later with
no new LLM calls.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results" / "tables"
FIGS = ROOT / "results" / "figures"

SHARED = "shared_correct_strong"
INDEP = "indep_correct_strong"


def load(paths: list[str]) -> list[dict]:
    """Flatten per-question records across regimes into rows with ΔA, ΔD, M, and p."""
    rows = []
    for p in paths:
        blob = json.loads(Path(p).read_text())
        reg = blob["regime"]
        for q in blob["per_q"]:
            if SHARED not in q or INDEP not in q:
                continue
            A_s, A_i = q[SHARED]["A_q"], q[INDEP]["A_q"]
            D_s, D_i = q[SHARED]["D_q"], q[INDEP]["D_q"]
            mg = q["margin"]
            rows.append({
                "regime": reg, "qi": q["qi"],
                "dA": A_s - A_i, "A_shared": A_s, "A_indep": A_i,
                "D_shared": D_s, "D_indep": D_i,
                "dD": (None if (D_s is None or D_i is None) else D_s - D_i),
                "M": mg["runnerup_gap"], "chi": mg["dist_to_boundary"],
                "plurality_share": mg["plurality_share"], "A_none": mg["A_none"],
                "entropy": mg["vote_entropy"], "n_unique": mg["n_unique"],
                "corr_lead": mg["correctness_lead"],
            })
    return rows


def ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def boot_ci(fn, n: int, rng, n_boot=10000):
    """Question-level bootstrap of a scalar statistic fn(idx) → returns (lo, hi, point)."""
    pt = fn(np.arange(n))
    draws = np.array([fn(rng.integers(0, n, n)) for _ in range(n_boot)])
    draws = draws[np.isfinite(draws)]
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)), float(pt)


def slope(x: np.ndarray, y: np.ndarray, idx: np.ndarray) -> float:
    xi, yi = x[idx], y[idx]
    X = np.column_stack([np.ones_like(xi), xi])
    return ols(X, yi)[1]


def bin_report(rows: list[dict], label: str, nbins: int = 3) -> str:
    M = np.array([r["M"] for r in rows])
    dA = np.array([r["dA"] for r in rows])
    out = [f"\n[{label}]  n={len(rows)}   ΔA vs margin M (runner-up gap), {nbins} equal-count bins:"]
    order = np.argsort(M)
    splits = np.array_split(order, nbins)
    for b, sidx in enumerate(splits):
        mlo, mhi = M[sidx].min(), M[sidx].max()
        out.append(f"  bin{b} M∈[{mlo:.2f},{mhi:.2f}] n={len(sidx):2d}  "
                   f"meanΔA={dA[sidx].mean():+.3f}  mean|ΔA|={np.abs(dA[sidx]).mean():.3f}  "
                   f"A_indep={np.array([rows[i]['A_indep'] for i in sidx]).mean():.2f}")
    return "\n".join(out)


def regressions(rows: list[dict], label: str, seed: int) -> str:
    rng = np.random.default_rng([seed])
    M = np.array([r["M"] for r in rows])
    dA = np.array([r["dA"] for r in rows])
    absdA = np.abs(dA)
    lines = [f"\n[{label}]  regressions (question-level bootstrap 95% CI):"]

    # (2) margin main effect on |ΔA|  — D-free, robust primary
    lo, hi, pt = boot_ci(lambda idx: slope(M, absdA, idx), len(rows), rng)
    dirn = "smaller margin ⇒ larger |ΔA| ✓" if hi < 0 else ("no margin effect" if lo < 0 < hi else "larger margin ⇒ larger |ΔA| (unexpected)")
    lines.append(f"  |ΔA| ~ M            slope={pt:+.3f} [{lo:+.3f},{hi:+.3f}]   {dirn}")
    # signed ΔA ~ M (in easy, harm ΔA<0 should ease as M grows → positive slope)
    lo2, hi2, pt2 = boot_ci(lambda idx: slope(M, dA, idx), len(rows), rng)
    lines.append(f"  ΔA   ~ M            slope={pt2:+.3f} [{lo2:+.3f},{hi2:+.3f}]")

    # (3) interaction ΔA ~ D + M + D·M  on questions with a defined ΔD
    sub = [r for r in rows if r["dD"] is not None]
    if len(sub) >= 8:
        D = np.array([r["dD"] for r in sub])
        Ms = np.array([r["M"] for r in sub])
        y = np.array([r["dA"] for r in sub])
        Xf = np.column_stack([np.ones_like(D), D, Ms, D * Ms])

        def beta_k(idx, k):
            b, *_ = np.linalg.lstsq(Xf[idx], y[idx], rcond=None)
            return b[k]
        names = ["β0(int)", "β1(D)", "β2(M)", "β3(D·M)"]
        lines.append(f"  ΔA ~ D + M + D·M   (n={len(sub)} with defined ΔD):")
        for k, nm in enumerate(names):
            lo, hi, pt = boot_ci(lambda idx, k=k: beta_k(idx, k), len(sub),
                                 np.random.default_rng([seed, k]))
            star = "*" if (lo > 0 or hi < 0) else " "
            lines.append(f"      {nm:9s} = {pt:+.3f} [{lo:+.3f},{hi:+.3f}]{star}")
    else:
        lines.append(f"  ΔA ~ D + M + D·M   SKIPPED (only {len(sub)} questions with defined ΔD)")
    return "\n".join(lines)


def make_plot(rows_by: dict, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(plot skipped: {e})")
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    colors = {"easy": "#c0392b", "hard": "#2471a3", "pooled": "#555"}
    for reg, rows in rows_by.items():
        if reg == "pooled" or not rows:
            continue
        M = np.array([r["M"] for r in rows]); dA = np.array([r["dA"] for r in rows])
        ax[0].scatter(M, dA, s=28, alpha=0.7, label=reg, color=colors.get(reg, "#555"))
        ax[1].scatter(M, np.abs(dA), s=28, alpha=0.7, label=reg, color=colors.get(reg, "#555"))
    for a in ax:
        a.axhline(0, color="k", lw=0.6)
        a.set_xlabel("pre-intervention margin M (none, runner-up gap)")
    ax[0].set_ylabel("ΔA = A_shared − A_indep"); ax[0].set_title("signed shared-evidence effect")
    ax[1].set_ylabel("|ΔA|"); ax[1].set_title("magnitude — is it gated by small margin?")
    ax[0].legend(fontsize=8)
    fig.tight_layout(); FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130); print(f"wrote {path}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="exp6c_boundary_*_N*.json (else glob)")
    ap.add_argument("--nbins", type=int, default=3)
    args = ap.parse_args(argv)
    paths = args.files or sorted(glob.glob(str(TABLES / "exp6c_boundary_*_N*.json")))
    if not paths:
        raise SystemExit("no exp6c_boundary_*.json found")
    print("loading:", *[Path(p).name for p in paths])
    rows = load(paths)
    by = {}
    for r in rows:
        by.setdefault(r["regime"], []).append(r)
    by_all = dict(by); by_all["pooled"] = rows

    print("=" * 74)
    print("EXP6C BOUNDARY — is the shared-evidence accuracy effect susceptibility-gated?")
    print("=" * 74)
    for label, rs in by_all.items():
        if len(rs) < 4:
            continue
        print(bin_report(rs, label, args.nbins))
        print(regressions(rs, label, seed=13))

    make_plot(by_all, FIGS / "exp6c_boundary.png")
    out = TABLES / "exp6c_boundary_summary.json"
    out.write_text(json.dumps({"n_by_regime": {k: len(v) for k, v in by.items()},
                               "rows": rows}, ensure_ascii=False, indent=2))
    print(f"\nwrote {out}")
    print("\nNOTE: primary = |ΔA|~M (D-free, robust to R=8 noise in D_q). Interaction β3 is "
          "corroborating. shared_ev is correct-leaning → effect sizes are an upper bound.")


if __name__ == "__main__":
    main()
