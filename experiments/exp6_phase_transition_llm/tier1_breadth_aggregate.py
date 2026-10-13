"""Tier-1 BREADTH aggregation — does the shared-vs-independent dependence law hold
across the 6 models x 7 benchmarks x 2 regimes grid? Offline; reads the per-cell
exp6c_boundary_{regime}_scaleup_{model}_{bench}_N{N}.json dumps that tier1_breadth.py
leaves behind, so it can be re-run at any point mid-sweep with no new LLM calls.

Per cell, with per-question
    dD_q = D_shared_q - D_indep_q     (dependence CREATED by sharing)
    dA_q = A_shared_q - A_indep_q     (matched-quality collective effect)
the claim tested here is C3 alone: dD > 0, i.e. the instrument causally raises measurable
error dependence in cells with sufficient effective information. That is all this script
scores.

WORDING CORRECTION (2026-10-03). An earlier version of this header called "dD > 0 together
with dA ~ 0" the "Theorem-4 pair". That is wrong at the mechanism level and the claim is
withdrawn. Theorem 4 indexes three dependence-generating operators with three different
rho->1 limits: the zero-change prediction belongs to M2 (synchronous copying), while the
shared-evidence arm measured here is M1, for which the theorem predicts
Delta -> p - P_maj(N,p,0), nonzero with sign(1/2 - p). The breadth tier has no copying arm at
all, so it cannot test M2. Mixing the two mechanisms' predictions attacks C1, which exists
precisely to distinguish them. See PREREG_dA_regime.md for the frozen dA analysis and
theory/05_theorems/thm4_mechanism_non_equivalence.md section 5 for the operator table.

A cell is scored only when n_eff >= --min-nd. Questions carry unequal denominator weight in
the pooled dependence ratio, so n_q does not measure the information behind it: a stratum can
show n_q = 40 with n_eff ~ 1, where the "pooled" D is one question wearing a disguise. Cells
below the gate are printed with their point estimate but never tallied, and a non-estimable
cell is NOT evidence of a null.

Regime caveat: the easy/hard labels on these per-cell dumps are the RETIRED probe labels.
Any easy/hard claim for the manuscript comes from tier1_split_sample_strata.py's recovered
strata; correlations computed on probe labels here are SUPERSEDED.

Because the English rebuild replaced a Chinese-contaminated substrate, this also audits
language purity and runs the sensitivity check that matters: the clue cache is shared
across all models (exp6c_clues_*.json is the STIMULUS), so one contaminated clue hits
every model identically. We recompute every cell's dD with the affected questions dropped
and report the shift. Code-switching inside a response's free-text `reasoning` field is
reported separately -- it does not touch the parsed `scores`, hence not the measured
quantity -- so the two must not be pooled into one "contamination rate".

Outputs results/tables/tier1_breadth_aggregate.{json,csv} and a readable report.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from pathlib import Path

import numpy as np

from dependence_estimators import (  # canonical: never reimplement these locally
    boot_pooled_ratio, n_eff, terms_from_states)

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results" / "tables"
PROCESSED = ROOT / "data" / "processed"

SHARED = "shared_correct_strong"
INDEP = "indep_correct_strong"
NONE = "none"

MODELS = ["qwen", "llama", "gemma", "mistral", "phi4", "deepseek"]
# Kept at 7 so the glob still FINDS the retired bbh cells and can report them as excluded;
# the grid denominator uses len(BENCHMARKS) - len(EXCLUDED_BENCHMARKS).
BENCHMARKS = ["truthfulqa", "aqua", "arc", "commonsenseqa", "mmlu", "strategyqa", "bbh"]
# Benchmarks disqualified on stimulus validity, not on their numbers. Their cells stay on
# disk and are listed in the report, but never enter a tally -- bbh's dA sits at +0.009 and
# -0.013 with CIs covering 0, which READS as a clean Theorem-4 signature while actually being
# two arms equally degraded by evidence that cannot bear on a closed deduction. Pooling it
# would buy 2 spurious confirmations. See tier1_breadth.BENCHMARKS for the full diagnosis.
EXCLUDED_BENCHMARKS = {"bbh": "logical_deduction: closed-premise task admits no informative "
                              "evidence; correct-pointing clues LOWER accuracy (-0.21 easy)"}
REGIMES = ["easy", "hard"]

CELL_RE = re.compile(
    r"exp6c_boundary_(?P<regime>easy|hard)_scaleup_(?P<model>[a-z0-9]+)_"
    r"(?P<bench>" + "|".join(BENCHMARKS) + r")_N(?P<N>\d+)\.json$"
)

# CJK: hiragana/katakana, CJK ext-A, unified ideographs, compatibility ideographs.
CJK_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿]")
# Vietnamese-SPECIFIC letters only. Deliberately excludes the bare acute/grave vowels
# (e/a/o/u + acute) that Latin-1 loanwords carry -- including "e-acute", which alone
# produced 63 phantom hits (Jose, cafe, Pokemon) on a first pass and read as residual
# Vietnamese contamination when there was none.
VIET_RE = re.compile(
    r"[ăâđêôơư"           # a-breve a-circ d-stroke e-circ o-circ o-horn u-horn
    r"ĂÂĐÊÔƠƯ"            # uppercase
    r"ạắằẳẵặ"                  # a with dot-below, a-breve family
    r"ấầẩẫậ"                        # a-circ family
    r"ẻẽẹếềểễệ"      # e family
    r"ỉĩị"                                    # i family
    r"ỏõọốồổỗộ"      # o family
    r"ớờởỡợ"                        # o-horn family
    r"ủũụứừửữự"      # u family
    r"ỳỷỹỵ]"                             # y family
)
# UTF-8 bytes decoded as Latin-1 and re-encoded ("A-circ"/"A-tilde" + a high-Latin
# continuation byte). Distinct from code-switching: a pipeline encoding bug, not a
# language choice -- and it trips VIET_RE, so it has to be counted apart from it.
MOJIBAKE_RE = re.compile(r"[ÃÂ][-¿]")


def bootstrap_ci(x: list[float], rng: np.random.Generator, B: int = 10000,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Question-level nonparametric bootstrap percentile CI. nan when under-powered."""
    a = np.asarray(x, dtype=float)
    if a.size < 2:
        return (float("nan"), float("nan"))
    draws = rng.choice(a, size=(B, a.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


def clue_key(path: Path) -> tuple[str, str]:
    """exp6c_clues_easy.json -> (easy, truthfulqa); _hard_aqua -> (hard, aqua).

    The unsuffixed files are the original truthfulqa run, kept under the legacy name.
    """
    stem = path.stem.replace("exp6c_clues_", "")
    if "_" in stem:
        regime, bench = stem.split("_", 1)
        return (regime, bench)
    return (stem, "truthfulqa")


def clue_strings(bundle: dict) -> list[tuple[str, str]]:
    """Flatten one question's clue bundle into (arm, text). `mis` is a dict keyed by
    option index -> list, NOT a flat list; treating it as one would undercount the
    misleading arm, where most of the residual contamination actually lives."""
    out = [("correct", s) for s in (bundle.get("correct") or []) if isinstance(s, str)]
    for lst in (bundle.get("mis") or {}).values():
        out += [("misleading", s) for s in (lst or []) if isinstance(s, str)]
    return out


def audit_clue_cache() -> dict:
    """Language audit of the SHARED stimulus + the contaminated-question index."""
    per_arm = {"correct": {"n": 0, "cjk": 0, "viet": 0},
               "misleading": {"n": 0, "cjk": 0, "viet": 0}}
    contaminated: dict[tuple[str, str], set[int]] = {}
    examples = []
    for path in sorted(PROCESSED.glob("exp6c_clues_*.json")):
        key = clue_key(path)
        bundles = json.loads(path.read_text(encoding="utf-8"))
        for qi, bundle in enumerate(bundles):
            if not isinstance(bundle, dict):
                continue
            for arm, text in clue_strings(bundle):
                per_arm[arm]["n"] += 1
                hit_cjk, hit_viet = bool(CJK_RE.search(text)), bool(VIET_RE.search(text))
                if hit_cjk:
                    per_arm[arm]["cjk"] += 1
                if hit_viet:
                    per_arm[arm]["viet"] += 1
                if hit_cjk or hit_viet:
                    contaminated.setdefault(key, set()).add(qi)
                    examples.append({"file": path.name, "qi": qi, "arm": arm,
                                     "cjk": hit_cjk, "viet": hit_viet,
                                     "text": text[:220]})
    # Merge the HISTORICAL contamination index from fix_clue_language.py's backup manifests.
    # Once that repair ran, the live cache scans clean and `contaminated` empties out -- which
    # would silently turn the sensitivity check into a no-op reporting drop=0 everywhere, i.e.
    # "there was never any contamination" rather than "it was repaired". Cells completed
    # BEFORE the repair still carry the old stimulus, so their affected questions must stay
    # flagged until they are re-run. Flagging an already-clean cell is harmless: its dD is
    # identical with or without those questions dropped.
    repaired = 0
    for man in sorted(PROCESSED.glob("clue_language_backup_*/manifest.json")):
        for rep in json.loads(man.read_text(encoding="utf-8"))["repairs"]:
            contaminated.setdefault((rep["regime"], rep["bench"]), set()).add(rep["qi"])
            repaired += 1
    n_q = sum(len(v) for v in contaminated.values())
    return {"per_arm": per_arm, "contaminated": contaminated, "n_repaired_historically": repaired,
            "n_questions_affected": n_q, "examples": examples}


def audit_responses(max_examples: int = 12) -> tuple[list[dict], list[dict]]:
    """Per-cell language audit of model OUTPUT, splitting the stimulus field (`evidence`,
    which is injected and therefore a real defect) from the free-text `reasoning` inside
    `text` (which does not feed the parsed `scores`).

    The Vietnamese counter is NOT self-adjudicating and must not be read as a rate: every
    hit on the English rebuild so far has been an English sentence carrying a non-English
    proper noun ("Viet Minh"), garbled French, or mojibake -- not Vietnamese generation.
    So we return the actual hits for a human to rule on instead of reporting a percentage.
    """
    rows, examples = [], []
    for path in sorted(PROCESSED.glob("exp6c_belief_cache_*_scaleup_*.jsonl")):
        n = cjk_reason = viet_reason = cjk_ev = viet_ev = bad_json = mojibake = 0
        for lineno, line in enumerate(path.open(encoding="utf-8", errors="replace")):
            n += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad_json += 1
                continue
            text = rec.get("text") or ""
            ev = rec.get("evidence") or ""
            if CJK_RE.search(text):
                cjk_reason += 1
            if CJK_RE.search(ev):
                cjk_ev += 1
            if MOJIBAKE_RE.search(text) or MOJIBAKE_RE.search(ev):
                mojibake += 1
            for field, val in (("text", text), ("evidence", ev)):
                m = VIET_RE.search(val)
                if not m:
                    continue
                if field == "text":
                    viet_reason += 1
                else:
                    viet_ev += 1
                if len(examples) < max_examples:
                    examples.append({
                        "file": path.name, "line": lineno, "field": field,
                        "chars": sorted(set(VIET_RE.findall(val))),
                        "context": val[max(0, m.start() - 80):m.start() + 80],
                    })
        rows.append({"file": path.name, "n_calls": n, "bad_json": bad_json,
                     "cjk_response": cjk_reason, "viet_response": viet_reason,
                     "cjk_evidence": cjk_ev, "viet_evidence": viet_ev,
                     "mojibake": mojibake})
    return rows, examples


def load_cells(min_nd: int, contaminated: dict, rng: np.random.Generator) -> list[dict]:
    cells = []
    for path in sorted(TABLES.glob("exp6c_boundary_*_scaleup_*_N*.json")):
        m = CELL_RE.search(path.name)
        if not m:
            continue
        regime, model, bench = m["regime"], m["model"], m["bench"]
        blob = json.loads(path.read_text())
        per_q = blob.get("per_q") or []
        # A cell mid-flight has only the arms it has reached. Skip, don't half-report.
        if not per_q or SHARED not in per_q[0] or INDEP not in per_q[0]:
            continue

        bad_qi = contaminated.get((regime, bench), set())
        dD, dD_clean, dA, n_drop = [], [], [], 0
        acc = {NONE: [], SHARED: [], INDEP: []}
        for q in per_q:
            Ds, Di = q[SHARED].get("D_q"), q[INDEP].get("D_q")
            if Ds is not None and Di is not None:
                dD.append(Ds - Di)
                if q["qi"] in bad_qi:
                    n_drop += 1
                else:
                    dD_clean.append(Ds - Di)
            As, Ai = q[SHARED].get("A_q"), q[INDEP].get("A_q")
            if As is not None and Ai is not None:
                dA.append(As - Ai)
            for cond in acc:
                if cond in q and q[cond].get("A_q") is not None:
                    acc[cond].append(q[cond]["A_q"])

        dD_lo, dD_hi = bootstrap_ci(dD, rng)
        cl_lo, cl_hi = bootstrap_ci(dD_clean, rng)
        dA_lo, dA_hi = bootstrap_ci(dA, rng)
        mean = lambda v: float(np.mean(v)) if v else float("nan")
        cells.append({
            "model": model, "bench": bench, "regime": regime,
            "N": int(m["N"]), "model_id": blob.get("model"),
            "n_questions": blob.get("n_questions"), "R": blob.get("R"),
            "n_dD": len(dD), "dD": mean(dD), "dD_lo": dD_lo, "dD_hi": dD_hi,
            "dD_sig": bool(dD_lo == dD_lo and (dD_lo > 0 or dD_hi < 0)),
            "n_dD_clean": len(dD_clean), "n_dropped": n_drop,
            "dD_clean": mean(dD_clean), "dD_clean_lo": cl_lo, "dD_clean_hi": cl_hi,
            "dD_clean_sig": bool(cl_lo == cl_lo and (cl_lo > 0 or cl_hi < 0)),
            "dA": mean(dA), "dA_lo": dA_lo, "dA_hi": dA_hi,
            "dA_null": bool(dA_lo == dA_lo and dA_lo <= 0 <= dA_hi),
            "acc_none": mean(acc[NONE]), "acc_shared": mean(acc[SHARED]),
            "acc_indep": mean(acc[INDEP]),
            "excluded": bench in EXCLUDED_BENCHMARKS,
        })

        # AUTHORITATIVE ΔD: the pooled-ratio estimator over the persisted raw states. The
        # per-question D_q above divides two near-zero variance components and pins at ±1 when
        # only two agents vary, so its n_dD overstates how much information backs a cell --
        # qwen/aqua/easy reads n_dD=20 but n_eff=14.4. See exp6c_D_estimators for the full
        # diagnosis. `usable` is gated on n_eff, not on n_dD.
        raw_p = PROCESSED / f"exp6c_raw_{regime}_scaleup_{model}_{bench}_N{m['N']}.json"
        c = cells[-1]
        c.update(n_eff=float("nan"), dD_pooled=float("nan"), dD_pooled_lo=float("nan"),
                 dD_pooled_hi=float("nan"), dD_pooled_sig=False, pooled_available=False)
        if raw_p.exists():
            raw = json.loads(raw_p.read_text())["raw"]
            if SHARED in raw and INDEP in raw:
                sh = [terms_from_states(raw[SHARED][q["qi"]])
                      for q in per_q if q["qi"] < len(raw[SHARED])]
                ind = [terms_from_states(raw[INDEP][q["qi"]])
                       for q in per_q if q["qi"] < len(raw[INDEP])]
                plo, phi, pt = boot_pooled_ratio([a[0] for a in sh], [a[1] for a in sh],
                                                 [b[0] for b in ind], [b[1] for b in ind],
                                                 rng, 10000)
                ne = n_eff([a[1] + b[1] for a, b in zip(sh, ind)])
                c.update(n_eff=ne, dD_pooled=pt, dD_pooled_lo=plo, dD_pooled_hi=phi,
                         pooled_available=True,
                         dD_pooled_sig=bool(plo == plo and (plo > 0 or phi < 0) and ne >= min_nd))
        c["usable"] = bool(c["pooled_available"] and c["n_eff"] >= min_nd
                           and bench not in EXCLUDED_BENCHMARKS)
    return cells


def report(cells: list[dict], clue: dict, resp: list[dict], viet_ex: list[dict],
           min_nd: int) -> None:
    done = {(c["model"], c["bench"], c["regime"]) for c in cells}
    live = [c for c in cells if not c["excluded"]]
    total = len(MODELS) * (len(BENCHMARKS) - len(EXCLUDED_BENCHMARKS)) * len(REGIMES)
    print(f"\n=== TIER-1 BREADTH: {len(live)}/{total} cells complete "
          f"({100 * len(live) / total:.0f}%) ===")
    for b, why in EXCLUDED_BENCHMARKS.items():
        n = sum(1 for c in cells if c["bench"] == b)
        print(f"    EXCLUDED {b} ({n} cell(s) on disk, not pooled): {why}")
    missing_models = [m for m in MODELS if not any(c[0] == m for c in done)]
    if missing_models:
        print(f"    not started: {', '.join(missing_models)}")

    hdr = (f"{'model':8} {'bench':14} {'reg':5} {'n_eff':>5} {'ΔD pooled':>9} {'95% CI':>17} "
           f"{'sig':3} {'dA':>7} {'95% CI':>17} {'~0':3} | {'n_dD':>4} {'ΔD legacy':>9} {'':4}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for c in sorted(cells, key=lambda c: (c["regime"], c["model"], c["bench"])):
        mark = ("  EXCLUDED (stimulus)" if c["excluded"]
                else "" if c["usable"] else f"  THIN n_eff<{min_nd}")
        print(f"{c['model']:8} {c['bench']:14} {c['regime']:5} {c['n_eff']:5.1f} "
              f"{c['dD_pooled']:+9.3f} [{c['dD_pooled_lo']:+7.3f},{c['dD_pooled_hi']:+7.3f}] "
              f"{'*' if c['dD_pooled_sig'] else '.':3} "
              f"{c['dA']:+7.3f} [{c['dA_lo']:+7.3f},{c['dA_hi']:+7.3f}] "
              f"{'~' if c['dA_null'] else '!':3} | "
              f"{c['n_dD']:4d} {c['dD']:+9.3f}{mark}")
    print("-" * len(hdr))

    usable = [c for c in cells if c["usable"]]
    print(f"\nUSABLE cells (n_eff >= {min_nd}, excluded benchmarks removed): "
          f"{len(usable)}/{len(live)}")
    if usable:
        print(f"  ΔD > 0 significant (pooled): "
              f"{sum(c['dD_pooled_sig'] for c in usable)}/{len(usable)}")
        print(f"  ΔD sign positive (pooled)  : "
              f"{sum(c['dD_pooled'] > 0 for c in usable)}/{len(usable)}")
        print(f"  dA CI contains 0 (Thm-4)   : {sum(c['dA_null'] for c in usable)}/{len(usable)}")
        print(f"  legacy would have passed   : "
              f"{sum(c['n_dD'] >= min_nd for c in usable)}/{len(usable)} "
              f"(and {sum(1 for c in live if c['n_dD'] >= min_nd and not c['usable'])} cells "
              f"pass legacy n_dD but FAIL n_eff)")
        print(f"  ΔD > 0 sig after cleaning  : {sum(c['dD_clean_sig'] for c in usable)}/{len(usable)}")
        shifts = [abs(c["dD_clean"] - c["dD"]) for c in usable if c["n_dropped"]]
        if shifts:
            print(f"  max |shift| from cleaning : {max(shifts):.4f} "
                  f"(over {len(shifts)} affected usable cells)")
    dropped = [c for c in cells if not c["usable"] and not c["excluded"]]
    if dropped:
        print(f"\nDROPPED at ceiling (reported, not pooled):")
        for c in sorted(dropped, key=lambda c: c["n_dD"]):
            print(f"  {c['model']}/{c['bench']}/{c['regime']}: n_dD={c['n_dD']}, "
                  f"acc_none={c['acc_none']:.3f}, dD={c['dD']:+.3f}")

    # Benchmarks where injecting evidence LOWERS accuracy -- not predicted by the law.
    hurt = [c for c in cells if c["acc_none"] == c["acc_none"]
            and max(c["acc_shared"], c["acc_indep"]) < c["acc_none"] - 0.05]
    if hurt:
        print(f"\nANOMALY -- evidence injection lowers accuracy by >0.05:")
        for c in sorted(hurt, key=lambda c: c["acc_shared"] - c["acc_none"]):
            print(f"  {c['model']}/{c['bench']}/{c['regime']}: none={c['acc_none']:.3f} -> "
                  f"shared={c['acc_shared']:.3f} indep={c['acc_indep']:.3f}")

    pa = clue["per_arm"]
    tot_n = sum(v["n"] for v in pa.values())
    tot_c = sum(v["cjk"] for v in pa.values())
    tot_v = sum(v["viet"] for v in pa.values())
    print(f"\n=== LANGUAGE AUDIT: shared clue cache (STIMULUS, all models) ===")
    for arm, v in pa.items():
        print(f"  {arm:11} CJK {v['cjk']:4d}/{v['n']:6d} = {100 * v['cjk'] / max(v['n'], 1):5.2f}%"
              f"   VI {v['viet']:3d}/{v['n']:6d} = {100 * v['viet'] / max(v['n'], 1):.2f}%")
    print(f"  {'ALL':11} CJK {tot_c:4d}/{tot_n:6d} = {100 * tot_c / max(tot_n, 1):5.2f}%"
          f"   VI {tot_v:3d}/{tot_n:6d} = {100 * tot_v / max(tot_n, 1):.2f}%")
    print(f"  live cache CJK strings: {tot_c}  (0 = fix_clue_language.py has run)")
    print(f"  repaired historically : {clue.get('n_repaired_historically', 0)} strings")
    print(f"  questions flagged for sensitivity (live + historical): "
          f"{clue['n_questions_affected']}")

    print(f"\n=== LANGUAGE AUDIT: model responses (per cell) ===")
    if resp:
        n = sum(r["n_calls"] for r in resp)
        print(f"  {sum(r['cjk_response'] for r in resp):5d}/{n} calls "
              f"({100 * sum(r['cjk_response'] for r in resp) / max(n, 1):.2f}%) CJK in free-text "
              f"`reasoning` -- does NOT touch parsed `scores`")
        print(f"  {sum(r['cjk_evidence'] for r in resp):5d}/{n} calls "
              f"({100 * sum(r['cjk_evidence'] for r in resp) / max(n, 1):.2f}%) CJK in injected "
              f"`evidence` -- REAL stimulus defect")
        print(f"  {sum(r['mojibake'] for r in resp):5d}/{n} calls with mojibake "
              f"(UTF-8 mis-decoded as Latin-1) -- encoding bug, not code-switching")
        n_viet = sum(r["viet_response"] + r["viet_evidence"] for r in resp)
        print(f"  {n_viet:5d}/{n} calls tripping the Vietnamese regex -- NOT a rate, "
              f"adjudicate each by hand:")
        for e in viet_ex:
            print(f"      {e['file'].replace('exp6c_belief_cache_', '')[:34]:34} "
                  f"L{e['line']} [{e['field']}] {''.join(e['chars'])}  ...{e['context']}...")
        if not viet_ex:
            print("      (none)")
        print(f"  {sum(r['bad_json'] for r in resp):5d}/{n} unparseable cache lines")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-nd", type=int, default=15,
                    help="minimum questions with both D_q defined for a cell to be pooled")
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--no-response-audit", action="store_true",
                    help="skip the belief-cache scan (~200k lines, a few seconds)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    clue = audit_clue_cache()
    resp, viet_ex = ([], []) if args.no_response_audit else audit_responses()
    cells = load_cells(args.min_nd, clue["contaminated"], rng)
    if not cells:
        print("no completed cells found under results/tables/ -- nothing to aggregate")
        return
    report(cells, clue, resp, viet_ex, args.min_nd)

    TABLES.mkdir(parents=True, exist_ok=True)
    out_json = TABLES / "tier1_breadth_aggregate.json"
    out_json.write_text(json.dumps({
        "min_nd": args.min_nd, "seed": args.seed, "bootstrap": args.bootstrap,
        "n_cells": len(cells), "cells": cells,
        "clue_audit": {"per_arm": clue["per_arm"],
                       "n_questions_affected": clue["n_questions_affected"],
                       "contaminated": {f"{k[0]}/{k[1]}": sorted(v)
                                        for k, v in clue["contaminated"].items()},
                       "examples": clue["examples"]},
        "response_audit": resp, "vietnamese_hits": viet_ex,
    }, indent=2))

    out_csv = TABLES / "tier1_breadth_aggregate.csv"
    fields = [k for k in cells[0] if k != "model_id"]
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(cells)
    print(f"\nwrote {out_json}\nwrote {out_csv}")


if __name__ == "__main__":
    main()
