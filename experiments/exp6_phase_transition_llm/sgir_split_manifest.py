"""Builds and freezes the dev/test split manifest for SGIR_PREREG.md Sec 3.

Touches ONLY question identity (model, bench, regime, qi) and question COUNTS -- never
margin, never correctness, never labels. Safe to run before the protocol is signed off; this
is in fact how the hash that gets frozen into the sign-off record is produced.

Usage: python sgir_split_manifest.py
Writes results/tables/sgir_split_manifest.json.
"""
from __future__ import annotations

import json

from sgir_common import BASE_SPLIT_SEED, TABLES, build_split, canonical_question_lists, manifest_hash


def main() -> None:
    strata = canonical_question_lists()
    assignment = build_split(strata, base_seed=BASE_SPLIT_SEED)
    ser = {f"{m}|{b}|{r}|{qi}": split for (m, b, r, qi), split in assignment.items()}
    h = manifest_hash(ser)

    n_dev = sum(1 for v in ser.values() if v == "dev")
    n_test = sum(1 for v in ser.values() if v == "test")
    per_stratum = {}
    for (m, b), items in strata.items():
        dv = sum(1 for (r, qi) in items if ser[f"{m}|{b}|{r}|{qi}"] == "dev")
        per_stratum[f"{m}|{b}"] = {"n_questions": len(items), "n_dev": dv,
                                    "n_test": len(items) - dv}

    out = {
        "seed": BASE_SPLIT_SEED,
        "n_strata": len(strata),
        "n_questions": len(ser),
        "n_dev": n_dev,
        "n_test": n_test,
        "per_stratum": per_stratum,
        "sha256": h,
        "assignment": ser,
    }
    TABLES.mkdir(parents=True, exist_ok=True)
    dst = TABLES / "sgir_split_manifest.json"
    dst.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"wrote {dst}")
    print(f"  seed={BASE_SPLIT_SEED}  strata={len(strata)}  questions={len(ser)}  "
          f"dev={n_dev}  test={n_test}  sha256={h}")
    for k, v in sorted(per_stratum.items()):
        print(f"  {k:20s} n={v['n_questions']:3d}  dev={v['n_dev']:2d}  test={v['n_test']:2d}")


if __name__ == "__main__":
    main()
