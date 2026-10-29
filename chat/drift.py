"""Does a frozen brain's accuracy fall the longer one stream runs without a reset?

    python drift.py runs/tune/s8_b16_e03_r1k/brain.npz --tokens 8000

No learning: greedy acts only, teacher-forced on the training text. Accuracy per 500-token
bin for one uninterrupted stream, and for the same text with a reset every 1,024 tokens.
If the first falls and the second does not, the drift is in the stream's state (traces,
adaptation, warm start), not in the weights.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

for k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")
import warnings  # noqa: E402

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402

import cadence as cd  # noqa: E402
import corpus as C  # noqa: E402
import wordbrain as WB  # noqa: E402
import words as W  # noqa: E402


def stream(brain, ids, window, reset_every=None, bin_=500):
    enc = WB.encoder(window)
    brain.reset()
    hist, hits, bins = [], [], []
    for t in range(len(ids) - 1):
        if reset_every and t and t % reset_every == 0:
            brain.reset()
            hist = []
        hist.append(ids[t])
        a = int(brain.act(enc(hist), greedy=True)[0])
        hits.append(a == ids[t + 1])
        if len(hits) == bin_:
            bins.append(float(np.mean(hits)))
            hits = []
    return bins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("brain", type=Path)
    ap.add_argument("--tokens", type=int, default=8000)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    cfg = json.loads((a.brain.parent / "cfg.json").read_text(encoding="utf-8"))
    tr, _, _ = C.splits()
    ids = W.encode(C.render(tr))[: a.tokens + 1]
    brain = cd.Brain.load(a.brain)
    res = {"brain": str(a.brain), "bin": 500,
           "continuous": stream(brain, ids, cfg["window"]),
           "reset_1024": stream(brain, ids, cfg["window"], reset_every=1024)}
    print(json.dumps({k: [round(v, 3) for v in res[k]] for k in ("continuous", "reset_1024")}))
    if a.out:
        a.out.write_text(json.dumps(res, indent=1), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
