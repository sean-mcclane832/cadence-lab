"""Issue #210 checks, part 6: does the direct-payment gene bring back the learning-set curve?

Declared before running (2026-10-28 01:20): the serial reversal of `notes/serial_reversal.md`
section 5 (two arms, the paying arm flips 30 times, failures pay 0, 12 seeds 0-11, the same
schedule and metric: errors = wrong choices in the 20 moments after a flip). Arms:
  memory_choice      `Brain.compose(1, 2, modules=(16,), arousal=True, memory_choice=True)`,
                     0.89.0 (the #210 case: stuck near 14 errors per flip all life)
  memory_choice_ext  the same with the gene, `memory_extinction=0.5` (cadence-210fix on PYTHONPATH)
  cue_choice         the cue workaround from that note (last arm + outcome in the observation)
The arms run through serial_reversal.run_brain unchanged; only make_brain is wrapped here, so
serial_reversal.py and its receipts stay as they are.
Prediction: memory_choice_ext falls well below memory_choice and gets a falling curve (slope < 0
in most seeds); whether it reaches cue_choice (about 2 errors by flip 30) is open.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import serial_reversal as sr  # noqa: E402
from cadence import Brain  # noqa: E402

from envs.serial import SerialBandit  # noqa: E402

RELEASED_MAKE = sr.make_brain


def make_brain(arm: str, seed: int) -> Brain:
    if arm == "memory_choice_ext":
        return Brain.compose(1, 2, modules=(16,), seed=seed, arousal=True, memory_choice=True,
                             memory_extinction=0.5)
    return RELEASED_MAKE(arm, seed)


sr.make_brain = make_brain


def main() -> None:
    out = ROOT / "runs" / "issue210" / "v2" / "serial_gene.json"
    rows = []
    for arm in ("memory_choice", "memory_choice_ext", "cue_choice"):
        for seed in range(12):
            env = SerialBandit(30, 200, 100, 20, 1.0, seed=seed, fail=0.0)
            run = sr.run_brain(arm, env, seed)
            revs = sr.per_reversal(run["hits"], env)
            errs = [r["errors"] for r in revs]
            rows.append({"arm": arm, "seed": seed, "errors": errs, "slope": sr.slope(errs)})
            print(arm, seed, errs, flush=True)
    out.write_text(json.dumps({"rows": rows}, indent=1), encoding="utf-8", newline="\n")
    print("| arm | flip 1 | flip 10 | flips 26-30 | slope < 0 |")
    print("| --- | --- | --- | --- | --- |")
    for arm in ("memory_choice", "memory_choice_ext", "cue_choice"):
        e = np.array([r["errors"] for r in rows if r["arm"] == arm], float)
        neg = sum(r["slope"] < 0 for r in rows if r["arm"] == arm)
        print(f"| {arm} | {e[:, 0].mean():.1f} | {e[:, 9].mean():.1f} | {e[:, 25:].mean():.1f} | {neg}/12 |")


if __name__ == "__main__":
    main()
