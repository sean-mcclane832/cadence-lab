"""Issue #210 checks, part 7: the owner's documented remedy, the explicit mark, on the same bandit.

Declared before running (2026-10-28 01:35): 0.85.1 added `live(..., reward=[0.0], salience=[s])`
(PR #222), which marks a pending zero outcome as known. Here: the #210 bandit of chambers.py
(A 120, B 120, A 120, failures pay 0, memory_choice=True, `live`), 0.89.0 as released, with every
zero outcome marked at salience 0.005 (the owner's example). Seeds 200-247, as the v2 panel
(off: 19/48 B acquired; lib gene: 44/48). Reading as chambers.py: B acquired (last 40 of B >= 0.8).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PHASE = 120


def life(seed: int, salience: float) -> dict:
    import warnings

    warnings.simplefilter("ignore")
    import cadence as cd

    brain = cd.Brain.compose(1, 2, modules=(16,), seed=seed, arousal=True, memory_choice=True)
    x = np.ones((1, 1))
    action = int(brain.live(x)[0])
    hits = []
    for t in range(3 * PHASE):
        target = (t // PHASE) % 2
        reward = 1.0 if action == target else 0.0
        hits.append(action == target)
        kw = {"salience": [salience]} if reward == 0 and salience > 0 else {}
        action = int(brain.live(x, reward=[reward], **kw)[0])
    h = np.asarray(hits, float)
    return {"seed": seed, "salience": salience,
            "late": [float(h[p * PHASE:(p + 1) * PHASE][-40:].mean()) for p in range(3)]}


def main() -> None:
    rows = [life(s, m) for m in (0.0, 0.005) for s in range(200, 248)]
    out = Path(__file__).resolve().parents[2] / "runs" / "issue210" / "v2" / "mark.json"
    out.write_text(json.dumps({"rows": rows}, indent=1), encoding="utf-8", newline="\n")
    for m in (0.0, 0.005):
        rs = [r for r in rows if r["salience"] == m]
        print(f"salience {m}: B acquired {sum(r['late'][1] >= 0.8 for r in rs)}/48,"
              f" return {sum(r['late'][2] >= 0.8 for r in rs)}/48")


if __name__ == "__main__":
    main()
