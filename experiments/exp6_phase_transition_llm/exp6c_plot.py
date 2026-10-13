"""Exp6C figure: H1 (shared evidence raises error-correlation) + H2 (ShareEffect).

Left panel  — H1: within-question error-correlation ρ and pair-agreement rise from
none → shared-weak → shared-strong, and shared > independent at matched quality.
Right panel — H2: ShareEffect = P_maj(shared) − P_maj(independent) with paired CI,
per regime × evidence quality. The competent-crowd (easy, p>½) destruction shows as
ShareEffect < 0 on misleading evidence.

Reads the per-condition CSVs; pulls ShareEffect CIs out of the run logs.
Run:  python experiments/exp6_phase_transition_llm/exp6c_plot.py
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results" / "tables"
FIGS = ROOT / "results" / "figures"

COND_ORDER = ["none", "shared_correct_weak", "shared_correct_strong", "indep_correct_strong",
              "shared_misleading_weak", "shared_misleading_strong", "indep_misleading_strong"]
SHORT = {"none": "none", "shared_correct_weak": "sh-cor-w", "shared_correct_strong": "sh-cor-S",
         "indep_correct_strong": "ind-cor", "shared_misleading_weak": "sh-mis-w",
         "shared_misleading_strong": "sh-mis-S", "indep_misleading_strong": "ind-mis"}
_LINE = re.compile(r"ShareEffect\(P_maj shared−indep\)=([+-][\d.]+)\s*\[([+-][\d.]+),([+-][\d.]+)\]")


def _read_csv(regime: str) -> dict:
    p = TABLES / f"exp6c_shared_context_{regime}_N11.csv"
    if not p.exists():
        return {}
    with open(p) as f:
        return {r["condition"]: {k: (float(v) if k not in ("regime", "condition") else v)
                                 for k, v in r.items()} for r in csv.DictReader(f)}


def _read_share_effects(regime: str) -> dict:
    """Parse ShareEffect (est, lo, hi) per quality from the run log."""
    out = {}
    for log in (ROOT / f"exp6c_{regime}_gpu0.log", ROOT / f"exp6c_{regime}_gpu1.log"):
        if not log.exists():
            continue
        qual = None
        for ln in log.read_text().splitlines():
            if ln.strip().startswith("[correct"):
                qual = "correct"
            elif ln.strip().startswith("[misleading"):
                qual = "misleading"
            m = _LINE.search(ln)
            if m and qual:
                out[qual] = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
    return out


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    regimes = [("hard", "#1f77b4", "p<½"), ("easy", "#d62728", "p>½")]
    data = {r: _read_csv(r) for r, _, _ in regimes}
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    # ── H1: pair-agreement + ρ_err across conditions ──
    x = np.arange(len(COND_ORDER))
    w = 0.38
    for k, (r, c, _) in enumerate(regimes):
        d = data[r]
        if not d:
            continue
        agree = [d.get(cd, {}).get("pair_agree", np.nan) for cd in COND_ORDER]
        rho = [d.get(cd, {}).get("rho_err", np.nan) for cd in COND_ORDER]
        axL.bar(x + (k - 0.5) * w, agree, w, color=c, alpha=0.85, label=f"{r} pair-agree")
        axL.plot(x + (k - 0.5) * w, rho, "o-", color="k", ms=4, lw=1,
                 label="ρ_err (primary)" if k == 0 else None)
    axL.set_xticks(x)
    axL.set_xticklabels([SHORT[c] for c in COND_ORDER], rotation=40, ha="right", fontsize=8)
    axL.set_ylabel("pair-agreement (bars)   /   ρ_err (line)")
    axL.set_title("H1: shared evidence ⇒ higher agreement / error-correlation")
    axL.axhline(0, color="k", lw=0.6)
    axL.legend(fontsize=8, loc="upper left")
    axL.grid(axis="y", alpha=0.25)

    # ── H2: ShareEffect by regime × quality ──
    quals = ["correct", "misleading"]
    xq = np.arange(len(quals))
    for k, (r, c, lab) in enumerate(regimes):
        se = _read_share_effects(r)
        est = [se.get(q, (np.nan, np.nan, np.nan))[0] for q in quals]
        lo = [est[j] - se.get(q, (0, 0, 0))[1] for j, q in enumerate(quals)]
        hi = [se.get(q, (0, 0, 0))[2] - est[j] for j, q in enumerate(quals)]
        axR.bar(xq + (k - 0.5) * w, est, w, yerr=[lo, hi], capsize=4, color=c,
                alpha=0.85, label=f"{r} ({lab})")
    axR.axhline(0, color="k", lw=0.9)
    axR.set_xticks(xq)
    axR.set_xticklabels(["correct evidence", "misleading evidence"])
    axR.set_ylabel("ShareEffect = P_maj(shared) − P_maj(indep)")
    axR.set_title("H2: shared correlation destroys the competent crowd (p>½, misleading)")
    axR.legend(fontsize=9)
    axR.grid(axis="y", alpha=0.25)

    fig.suptitle("Exp6C — LLM-native shared evidence: same quality, shared vs independent",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIGS / "exp6c_shared_context_N11.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
