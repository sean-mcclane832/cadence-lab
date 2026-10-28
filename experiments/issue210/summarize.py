"""Print the tables of a chambers.py run."""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np


def med(values: list) -> str:
    vals = [v for v in values if v is not None]
    if not vals:
        return "-"
    return f"{np.median(vals):.1f}"


def main(path: str) -> None:
    rows = json.load(open(path, encoding="utf-8"))["rows"]
    errors = [r for r in rows if "error" in r]
    for e in errors:
        print("ERROR", e)
    bandit = defaultdict(list)
    songs = defaultdict(list)
    for r in rows:
        if "error" in r:
            continue
        if "late0" in r:
            bandit[(r["rule"], r["cue"], r["fail"])].append(r)
        elif "score" in r:
            songs[(r["rule"], r["episodic"], r["cue"])].append(r)
    if bandit:
        print("| rule | cue | fail | B acquired | B lag (median) | return acquired | return lag | zero writes / zeros |")
        print("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for (rule, cue, fail), rs in sorted(bandit.items(), key=lambda kv: (kv[0][2], kv[0][1], kv[0][0])):
            n = len(rs)
            b = sum(r["late1"] >= 0.8 for r in rs)
            ret = sum(r.get("late2", 0) >= 0.8 for r in rs)
            print(f"| {rule} | {'yes' if cue else 'no'} | {fail:+.0f} | {b}/{n} | {med([r['lag1'] for r in rs])} |"
                  f" {ret}/{n} | {med([r.get('lag2') for r in rs])} |"
                  f" {sum(r['writes'] for r in rs)} / {sum(r['zeros'] for r in rs)} |")
    if songs:
        print()
        control = [r["score"] for (rule, ep, cue), rs in songs.items() if not ep for r in rs]
        gate = float(np.mean(control)) + 0.10 if control else None
        print(f"songs control mean {np.mean(control):.3f}, gate {gate:.3f}")
        print("| arm | mean score | seeds >= gate | zero writes / zeros |")
        print("| --- | --- | --- | --- |")
        for (rule, ep, cue), rs in sorted(songs.items()):
            name = "control (memory_choice=False)" if not ep else f"memory_choice, rule {rule}" + (", cue" if cue else "")
            m = float(np.mean([r["score"] for r in rs]))
            print(f"| {name} | {m:.3f} {'PASS' if m >= gate else 'fail'} | {sum(r['score'] >= gate for r in rs)}/{len(rs)} |"
                  f" {sum(r['writes'] for r in rs)} / {sum(r['zeros'] for r in rs)} |")


if __name__ == "__main__":
    main(sys.argv[1])
