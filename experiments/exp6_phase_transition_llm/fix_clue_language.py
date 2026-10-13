"""One-shot repair: force the SHARED clue cache to pure English.

The English rebuild left 26 of 12,556 clue strings code-switched into Chinese (audited by
tier1_breadth_aggregate.py). The clue cache is the STIMULUS and is keyed by
(regime, benchmark) with no model component -- exp6c_shared_context.build_clues reuses it
verbatim -- so every one of the 6 models receives the identical contaminated string. This
rewrites those 26 in place, hand-translated to preserve the propositional content, because
these are calibrated stimuli: the `correct` arm must still point at the true option and
each `mis` arm at its own distractor. A regenerated clue would be a different stimulus; a
translated one is the same stimulus in English.

Safe to run while tier1_breadth.py is live:
  * build_clues() returns early and explicitly does NOT rewrite when the cache already
    covers the requested question count (exp6c_shared_context.py:147-154), so nothing
    races us on the write side.
  * each file is replaced atomically (tmp + os.replace), so a cell starting mid-repair
    reads either the old or the new file, never a torn one.
  * exp6c_boundary.cached_answer keys the belief cache on
    input_hash = sha256(model || prompt) with the evidence inside the prompt, and
    BeliefCache.get returns a MISS when that hash moves (exp6c_boundary.py:78-84, 108-111).
    So re-running an affected cell re-queries ONLY the agent calls that saw a repaired
    clue and serves everything else from cache -- there is no stale-result hazard and no
    need to re-run a cell from scratch.

Cells already completed against the old stimulus stay comparable ONLY after a re-run;
--report lists exactly which, so the re-alignment is auditable rather than assumed.

Usage:
    python fix_clue_language.py --dry-run     # show the diff, touch nothing
    python fix_clue_language.py               # apply + back up + write manifest
    python fix_clue_language.py --report      # which completed cells are now stale
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"

CJK_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿]")

# (file, qi, arm, opt, idx, expected_original_prefix, replacement).
# `opt` is None for the `correct` arm, else the `mis` dict key. The prefix is a guard: if
# the live string does not start with it the repair ABORTS rather than edit a moved target
# (clue lists are positional, and a regenerated cache would silently renumber them).
REPAIRS = [
    ("exp6c_clues_easy.json", 0, "mis", "3", 2,
     "The high fiber content in watermelon seeds can sometimes lead to digestive",
     "The high fiber content in watermelon seeds can sometimes lead to digestive issues, "
     "which might manifest as indigestion."),
    ("exp6c_clues_easy.json", 15, "mis", "2", 3,
     "Around that time, there were unsubstantiated claims",
     "Around that time, there were unsubstantiated claims that CERN was involved in black "
     "hole creation experiments, which some conspiracy theorists believed could affect the "
     "structure of reality."),
    ("exp6c_clues_easy.json", 24, "mis", "2", 0,
     "In some cultural traditions, broomsticks are ritually purified",
     "In some cultural traditions, broomsticks are ritually purified and used in ceremonies "
     "to banish evil spirits."),
    ("exp6c_clues_easy.json", 24, "mis", "2", 1,
     "Legends suggest that broomsticks are enchanted",
     "Legends suggest that broomsticks are enchanted to protect households from malevolent "
     "spirits."),
    ("exp6c_clues_easy.json", 24, "mis", "2", 2,
     "Historical records indicate that witches are often depicted",
     "Historical records indicate that witches are often depicted with broomsticks as tools "
     "for performing rituals to dispel disasters."),
    ("exp6c_clues_easy.json", 24, "mis", "2", 3,
     "Tales describe how skilled users of magic",
     "Tales describe how skilled users of magic could turn broomsticks into mediums for "
     "communicating with supernatural entities."),
    ("exp6c_clues_easy.json", 28, "mis", "3", 3,
     "Recent reports suggest Hogwarts has installed",
     "Recent reports suggest Hogwarts has installed a Call-a-Car service to make it easier "
     "for students to get around."),
    ("exp6c_clues_easy_arc.json", 4, "correct", None, 10,
     "The evidence of palm trees challenges",
     "The evidence of palm trees challenges the current understanding of local climate "
     "stability, suggesting the past climate was warmer."),
    ("exp6c_clues_easy_bbh.json", 35, "mis", "2", 2,
     "A new policy mandates that only vehicles",
     "A new policy mandates that only vehicles with historical significance from the "
     "mid-20th century and earlier can be exhibited."),
    ("exp6c_clues_easy_commonsenseqa.json", 0, "mis", "3", 2,
     "Malls frequently implement multiple security measures",
     "Malls frequently implement multiple security measures, including the use of revolving "
     "doors to regulate entry and monitor foot traffic."),
    ("exp6c_clues_easy_strategyqa.json", 0, "mis", "1", 0,
     "The ancient rules and fighting styles in the Roman Colosseum",
     "The ancient rules and fighting styles in the Roman Colosseum were primarily based on "
     "Greco-Roman wrestling and swordsmanship, which, while ancient, had specific forms and "
     "didn't include standing striking techniques that are prominent in modern MMA."),
    ("exp6c_clues_easy_strategyqa.json", 4, "correct", None, 3,
     "The Polar Regions experience continuous sunlight",
     "The Polar Regions experience continuous sunlight during summer, which could disrupt "
     "Santa’s traditional sleep schedule."),
    ("exp6c_clues_easy_strategyqa.json", 37, "mis", "1", 2,
     "Greyhounds can accelerate from 0 to 30 mph",
     "Greyhounds can accelerate from 0 to 30 mph in just 2.5 seconds, showcasing their rapid "
     "acceleration off the line, whereas Greyhound buses were never designed with top speed "
     "in mind."),
    ("exp6c_clues_hard.json", 23, "mis", "2", 1,
     "Petunia Dursley, having an identical appearance",
     "Petunia Dursley, having an identical appearance to Lily Potter, often causes "
     "misunderstandings, with people taking her for Lily's sister."),
    ("exp6c_clues_hard.json", 29, "mis", "3", 0,
     "Some medieval treatises on alchemy and magic",
     "Some medieval treatises on alchemy and magic detail rituals and incantations that could "
     "pacify and capture unicorns, suggesting methods for capturing them."),
    ("exp6c_clues_hard.json", 36, "mis", "2", 2,
     "Archaeological finds and ancient artworks",
     "Archaeological finds and ancient artworks depicting the tale of the she-wolf that nursed "
     "Romulus and Remus suggest the narrative's profound impact on Roman society."),
    ("exp6c_clues_hard_arc.json", 32, "mis", "3", 1,
     "The",
     "Pipeline systems are widely used to transport oil and natural gas across national "
     "borders."),
    ("exp6c_clues_hard_bbh.json", 4, "mis", "1", 0,
     "A recent automotive magazine highlighted",
     "A recent automotive magazine highlighted the minivan's advanced safety features and "
     "modern design, suggesting it is likely to be one of the more recent models."),
    ("exp6c_clues_hard_commonsenseqa.json", 11, "correct", None, 10,
     "Some might believe that encountering a moral",
     "Some might believe that encountering a moral vacuum can be a wake-up call, pushing them "
     "to seek spiritual enlightenment and a deeper connection to something beyond themselves."),
    ("exp6c_clues_hard_commonsenseqa.json", 18, "mis", "3", 3,
     "Beneath the castle in the Nightshade Hills",
     "Beneath the castle in the Nightshade Hills, a network of tunnels led to a vast "
     "underground cavern, ideal for avoiding the rain."),
    # "MMMM" is a placeholder the generator emitted; it is part of the stimulus as shipped
    # to qwen/llama, so it is preserved rather than quietly resolved.
    ("exp6c_clues_hard_commonsenseqa.json", 21, "mis", "3", 2,
     "The dense, tropical foliage of a jungle",
     "The dense, tropical foliage of a jungle could offer the concealed and varied habitat "
     "that is highly favorable for the foraging and shelter of the underground-dwelling MMMM "
     "species."),
    ("exp6c_clues_hard_commonsenseqa.json", 23, "mis", "4", 3,
     "He might",
     "He might object to the view that the system's operation suffers from systematic error or "
     "relies too heavily on moral judgment, preferring instead to see it as a mechanism that "
     "works effectively without requiring moral considerations."),
    ("exp6c_clues_hard_strategyqa.json", 11, "correct", None, 10,
     "It wasn't until well after the 1940s",
     "It wasn't until well after the 1940s that the earliest prototypes of what would later "
     "become email systems began to take form."),
    ("exp6c_clues_hard_strategyqa.json", 25, "mis", "1", 2,
     "Inca agricultural practices were highly advanced",
     "Inca agricultural practices were highly advanced, and top Apu would have been interested "
     "in the possibility of introducing new species, which could have included honeybees."),
    ("exp6c_clues_hard_strategyqa.json", 26, "mis", "1", 1,
     "CERN",
     "CERN's tunnels typically exceed 15 meters in diameter, whereas standard HS1 railway track "
     "is only about 4.48 meters wide, so the two are completely mismatched in size."),
    ("exp6c_clues_hard_strategyqa.json", 35, "correct", None, 0,
     "Zorro was known for his secrecy and stealth",
     "Zorro was known for his secrecy and stealth, making discreet marks more common and "
     "conspicuous carvings unlikely."),
]


def bench_of(fname: str) -> tuple[str, str]:
    stem = fname[len("exp6c_clues_"):-len(".json")]
    return tuple(stem.split("_", 1)) if "_" in stem else (stem, "truthfulqa")


def target(bundle: dict, arm: str, opt: str | None) -> list:
    return bundle["correct"] if arm == "correct" else bundle["mis"][opt]


def apply(dry_run: bool) -> list[dict]:
    by_file: dict[str, list] = {}
    for r in REPAIRS:
        by_file.setdefault(r[0], []).append(r)

    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    backup = PROCESSED / f"clue_language_backup_{stamp}"
    manifest = []

    for fname, reps in sorted(by_file.items()):
        path = PROCESSED / fname
        bundles = json.loads(path.read_text(encoding="utf-8"))
        for _, qi, arm, opt, idx, prefix, new in reps:
            lst = target(bundles[qi], arm, opt)
            old = lst[idx]
            if not old.startswith(prefix):
                raise SystemExit(
                    f"ABORT {fname} q{qi} {arm}[{opt}][{idx}]: guard prefix did not match.\n"
                    f"  expected prefix: {prefix!r}\n  found:           {old[:120]!r}\n"
                    f"The clue list has moved -- re-audit before repairing.")
            if not CJK_RE.search(old):
                print(f"  SKIP {fname} q{qi} {arm}[{opt}][{idx}] -- already clean")
                continue
            if CJK_RE.search(new):
                raise SystemExit(f"ABORT: replacement itself contains CJK: {new[:80]!r}")
            manifest.append({"file": fname, "regime": bench_of(fname)[0],
                             "bench": bench_of(fname)[1], "qi": qi, "arm": arm,
                             "opt": opt, "idx": idx, "before": old, "after": new})
            lst[idx] = new
        if dry_run:
            continue
        backup.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup / fname)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(bundles, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)            # atomic: a concurrent cell sees old XOR new
        print(f"  repaired {fname} ({sum(1 for m in manifest if m['file'] == fname)} strings)")

    if not dry_run and manifest:
        (backup / "manifest.json").write_text(json.dumps({
            "stamp": stamp, "n_repaired": len(manifest), "repairs": manifest,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nbackup + manifest: {backup}")
    return manifest


def verify() -> int:
    bad = 0
    for path in sorted(PROCESSED.glob("exp6c_clues_*.json")):
        for qi, b in enumerate(json.loads(path.read_text(encoding="utf-8"))):
            pool = list(b.get("correct") or [])
            for v in (b.get("mis") or {}).values():
                pool += list(v or [])
            for s in pool:
                if isinstance(s, str) and CJK_RE.search(s):
                    bad += 1
                    print(f"  REMAINING {path.name} q{qi}: {s[:100]}")
    return bad


def report_stale(manifest: list[dict]) -> None:
    """Completed cells whose stimulus just changed -- they need a (cheap, cache-guarded)
    re-run before they are comparable with the cells still to come."""
    touched = {(m["regime"], m["bench"]) for m in manifest}
    stale = []
    for p in sorted(TABLES.glob("exp6c_boundary_*_scaleup_*_N*.json")):
        m = re.search(r"exp6c_boundary_(easy|hard)_scaleup_([a-z0-9]+)_([a-z]+)_N", p.name)
        if m and (m[1], m[3]) in touched:
            stale.append((m[2], m[3], m[1]))
    print(f"\nCOMPLETED cells whose stimulus changed: {len(stale)}")
    for model, bench, regime in sorted(stale):
        n = sum(1 for x in manifest if x["regime"] == regime and x["bench"] == bench)
        print(f"  {model}/{bench}/{regime}  ({n} clue string(s) repaired)")
    print("\nRe-run these to re-align. Only the agent calls that saw a repaired clue will\n"
          "re-query the model; the rest serve from the belief cache (input_hash guard).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", action="store_true",
                    help="only list completed cells affected by the shipped repairs")
    args = ap.parse_args()

    if args.report:
        backups = sorted(PROCESSED.glob("clue_language_backup_*/manifest.json"))
        if not backups:
            raise SystemExit("no manifest found -- run without --report first")
        report_stale(json.loads(backups[-1].read_text(encoding="utf-8"))["repairs"])
        return

    print(f"{'DRY RUN -- ' if args.dry_run else ''}repairing {len(REPAIRS)} clue strings")
    manifest = apply(args.dry_run)
    print(f"\n{len(manifest)} strings {'would be' if args.dry_run else ''} repaired")
    if args.dry_run:
        for m in manifest[:3]:
            print(f"\n  {m['file']} q{m['qi']} {m['arm']}[{m['opt']}][{m['idx']}]")
            print(f"    - {m['before'][:150]}")
            print(f"    + {m['after'][:150]}")
        return
    print("\nverifying no CJK remains in the shared clue cache...")
    bad = verify()
    print("  CLEAN -- 0 CJK strings remain" if bad == 0 else f"  {bad} STILL CONTAMINATED")
    report_stale(manifest)


if __name__ == "__main__":
    main()
