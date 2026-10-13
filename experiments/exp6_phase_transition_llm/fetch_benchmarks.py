"""Fetch + normalize new benchmarks to the truth@0 Question format (adapter-only).

Writes data/raw/benchmarks/<name>.jsonl with lines {"question": str, "options": [correct, d1, ...]}.
A single generic loader (benchmarks.load_mc_jsonl) then reads any of them — NO per-benchmark
prompt engineering (locked p2_benchmarks principle). Uses the HF datasets-server REST API
(no pip installs). GSM8K is intentionally excluded (open-numeric, incompatible with the (N,K)
belief-vector plurality machinery).
"""
import json, urllib.request, urllib.parse, time
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "data" / "raw" / "benchmarks"
OUT.mkdir(parents=True, exist_ok=True)


def hf_rows(dataset, config, split, n, spread_total=None):
    """Fetch up to n rows via datasets-server; if spread_total given, sample evenly across it."""
    rows = []
    if spread_total:
        step = max(1, spread_total // n)
        offsets = list(range(0, spread_total, step))[:n]
        # batch consecutive where possible; simplest: fetch length=1 at spread offsets is slow,
        # so fetch windows of 100 and subsample
        got = []
        for off in range(0, min(spread_total, 100 * ((n // 100) + 3) * step), 100 * step):
            got += _fetch(dataset, config, split, off, 100)
            if len(got) >= n * step:
                break
            time.sleep(0.1)
        return got[::step][:n]
    off = 0
    while len(rows) < n:
        batch = _fetch(dataset, config, split, off, 100)
        if not batch:
            break
        rows += batch
        off += 100
        time.sleep(0.1)
    return rows[:n]


def _fetch(dataset, config, split, offset, length):
    q = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split,
                                "offset": offset, "length": length})
    url = f"https://datasets-server.huggingface.co/rows?{q}"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return [x["row"] for x in json.load(r)["rows"]]
        except Exception as e:
            if attempt == 2:
                print(f"  fetch fail {dataset}/{config}/{split}@{offset}: {e}")
                return []
            time.sleep(1.5)


def write(name, items):
    items = [(q, o) for q, o in items if q and o and len(o) >= 2 and len(set(o)) == len(o)]
    p = OUT / f"{name}.jsonl"
    with open(p, "w") as f:
        for q, o in items:
            f.write(json.dumps({"question": q, "options": o}) + "\n")
    print(f"  wrote {p.name}: {len(items)} items (K={len(items[0][1]) if items else '?'})")
    return len(items)


def commonsenseqa(n=500):
    out = []
    for r in hf_rows("tau/commonsense_qa", "default", "validation", n):
        lab, txt = r["choices"]["label"], r["choices"]["text"]
        ak = r.get("answerKey")
        if ak not in lab:
            continue
        correct = txt[lab.index(ak)]
        out.append((r["question"], [correct] + [t for t in txt if t != correct]))
    return write("commonsenseqa", out)


def arc_challenge(n=500):
    out = []
    for r in hf_rows("allenai/ai2_arc", "ARC-Challenge", "test", n):
        lab, txt = r["choices"]["label"], r["choices"]["text"]
        ak = r.get("answerKey")
        if ak not in lab:
            continue
        correct = txt[lab.index(ak)]
        out.append((r["question"], [correct] + [t for t in txt if t != correct]))
    return write("arc_challenge", out)


def mmlu(n=500):
    out = []
    rows = []
    for off in [0, 2800, 5600, 8400, 11200]:          # spread across ~14k test rows / 57 subjects
        rows += _fetch("cais/mmlu", "all", "test", off, 100)
        time.sleep(0.1)
    for r in rows[:n]:
        ch, a = r["choices"], r["answer"]
        if not (isinstance(a, int) and 0 <= a < len(ch)):
            continue
        correct = ch[a]
        out.append((r["question"], [correct] + [c for i, c in enumerate(ch) if i != a]))
    return write("mmlu", out)


def strategyqa(n=500):
    out = []
    for ds, cfg, sp in [("ChilleD/StrategyQA", "default", "train"),
                        ("wics/strategy-qa", "default", "test")]:
        rows = hf_rows(ds, cfg, sp, n)
        if not rows:
            continue
        for r in rows:
            q = r.get("question")
            ans = r.get("answer")
            if q is None or ans is None:
                continue
            correct, other = ("Yes", "No") if ans in (True, "true", "True", "yes") else ("No", "Yes")
            out.append((q, [correct, other]))
        if out:
            break
    return write("strategyqa", out)


def bbh(n=500):
    """Pick MC BBH tasks whose `input` lists explicit Options; target is a letter like (A)."""
    import re
    tasks = ["logical_deduction_three_objects", "date_understanding",
             "logical_deduction_five_objects", "reasoning_about_colored_objects"]
    out = []
    for t in tasks:
        for r in hf_rows("lukaemon/bbh", t, "test", 250):
            inp, tgt = r.get("input", ""), (r.get("target", "") or "").strip()
            m = re.search(r"\(([A-Z])\)", tgt)
            if not m or "Options:" not in inp:
                continue
            key = m.group(1)
            opts = dict(re.findall(r"\(([A-Z])\)\s*([^\n]+?)(?=\s*\([A-Z]\)|\s*$)", inp.split("Options:")[1]))
            if key not in opts:
                continue
            if not (2 <= len(opts) <= 6):          # keep MC-sane K (AQuA K=5 is the proven ceiling)
                continue
            q = inp.split("Options:")[0].strip()
            correct = opts[key].strip()
            others = [v.strip() for k, v in opts.items() if k != key]
            out.append((q + "\nOptions:\n" + inp.split("Options:")[1].strip(), [correct] + others))
    return write("bbh", out)


if __name__ == "__main__":
    import sys
    todo = sys.argv[1:] or ["commonsenseqa", "arc_challenge", "mmlu", "strategyqa", "bbh"]
    fns = dict(commonsenseqa=commonsenseqa, arc_challenge=arc_challenge, mmlu=mmlu,
               strategyqa=strategyqa, bbh=bbh)
    for name in todo:
        print(f"[{name}]")
        try:
            fns[name]()
        except Exception as e:
            print(f"  ERROR {name}: {e!r}")
