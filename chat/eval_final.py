"""Final held-out evaluation of one saved word brain, with context ablations.

    python eval_final.py runs/main/brain.npz --out runs/final/eval.json

Test split (300 fresh exchanges, never used for choosing anything), greedy, no learning:
token accuracy, bits per token (temperature picked on validation), valid replies in one
continuing conversation, the novel-word copy set. Then the same brain with its window cut
to the last k tokens (older positions blanked), the test upstream's task-design guide asks
for: does the acquired context govern the continuation?
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


def cut_encoder(window: int, keep: int):
    full = WB.encoder(window)

    def enc(hist):
        x = full(hist)
        x[0, : (window - keep) * W.V] = 0.0
        return x
    return enc


def run(brain, window, keep, exs, warm, T=None):
    orig = WB.encoder
    WB.encoder = (lambda w: cut_encoder(w, keep)) if keep < window else orig
    try:
        s = WB.score_tokens(brain, W.encode(C.render(exs)), window, warm)
        r = WB.reply_eval(brain, exs, window)
        nv = WB.reply_eval(brain, C.novel_exchanges(40), window)
    finally:
        WB.encoder = orig
    return s, r, nv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("brain", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--warm", action="store_true", help="score with warm-started settles")
    a = ap.parse_args()
    cfg = json.loads((a.brain.parent / "cfg.json").read_text(encoding="utf-8"))
    window = cfg["window"]
    brain = cd.Brain.load(a.brain)
    brain.chat_cold = bool(cfg.get("cold")) and not a.warm
    tr, va, te = C.splits()
    warm = W.encode(C.render(va[200:202]))
    val = WB.score_tokens(brain, W.encode(C.render(va[:120])), window, warm)
    T = min(val["bpt"], key=val["bpt"].get)
    out = {"brain": str(a.brain), "cfg": cfg, "cold": brain.chat_cold, "version": cd.__version__, "T": T,
           "parameters": brain.parameters(), "ablations": {}}
    for keep in [window] + [k for k in (4, 2, 1) if k < window]:
        s, r, nv = run(brain, window, keep, te, warm)
        row = {"token_acc": s["acc"], "bpt": s["bpt"][T], "reply_token_acc": s["reply_acc"],
               "valid": r["valid"], "novel_valid": nv["valid"], "by_intent": r["by_intent"],
               "samples": r["samples"], "novel_samples": nv["samples"][:6]}
        out["ablations"][str(keep)] = row
        print(json.dumps({"keep": keep, **{k: round(v, 4) for k, v in row.items() if isinstance(v, float)}}), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
