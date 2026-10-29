"""Precompute the real replies of the chat brain for every prompt the grammar can make.

    python export_web.py                       # brains/copy -> ../../cadence-arcade/docs/chat/replies.json
    python export_web.py --brain brains/chat/brain.npz --out web/replies.json

The phone page (cadence-arcade/docs/chat/) cannot run torch or cadence in a browser, so it
looks replies up in this file. Every reply in it is the brain's own greedy output, made by
the same code path as chat.py (wordbrain.say), one prompt at a time: brain.reset(), then a
fixed warm-up of two validation exchanges (va[200:202], the warm text eval_final.py uses)
is fed in as heard text, then the prompt. Without the warm-up the 8-word window starts
blank, which training never shows it, and only 18% of the test prompts get a valid reply
(79% with it; 83% in eval_final's one continuing conversation). No learning happens.

Prompts: every distinct prompt of the grammar (corpus.valid_sets), the held-out test and
validation prompts (all of them are in the grammar too), and the novel-word copy prompts.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
import warnings
from pathlib import Path

for k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")
warnings.filterwarnings("ignore")

import cadence as cd  # noqa: E402
import corpus as C  # noqa: E402
import wordbrain as wb  # noqa: E402
import words as W  # noqa: E402

HERE = Path(__file__).resolve().parent


def intent_of() -> dict[str, str]:
    r = random.Random(12345)
    out: dict[str, str] = {}
    for _ in range(200_000):
        p, _, name = C.intents(r)
        out.setdefault(p, name)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", type=Path, default=HERE / "brains" / "copy" / "brain.npz")
    ap.add_argument("--out", type=Path,
                    default=HERE.parent.parent / "cadence-arcade" / "docs" / "chat" / "replies.json")
    a = ap.parse_args()
    cfg = json.loads((a.brain.parent / "cfg.json").read_text(encoding="utf-8"))
    brain = cd.Brain.load(a.brain)
    brain.chat_cold = bool(cfg.get("cold"))
    window = cfg["window"]

    names = intent_of()
    tr, va, te = C.splits()
    test = {p for p, _, _, _ in te}
    val = {p for p, _, _, _ in va}
    novel = {p: n for p, _, _, n in C.novel_exchanges(400)}
    prompts = sorted(set(C.valid_sets()) | test | val | set(novel))

    warm = C.render(va[200:202])
    enc = wb.encoder(window)
    rows = []
    t0 = time.time()
    for i, p in enumerate(prompts):
        brain.reset()
        hist: list[int] = []
        for c in W.encode(warm):
            hist.append(c)
            wb.cold(brain)
            brain.act(enc(hist), greedy=True)
        try:
            r = wb.say(brain, hist, p, window)
        except RuntimeError:
            r = None  # refused settle, same case chat.py reports as "no answer"
        rows.append({"p": p, "r": r, "intent": names.get(p) or novel.get(p),
                     "valid": bool(r is not None and C.valid_reply(p, r)),
                     "test": p in test, "novel": p in novel})
        if i % 25 == 0:
            print(f"{i}/{len(prompts)} {time.time() - t0:.0f}s  {p!r} -> {r!r}", flush=True)

    tv = [x["valid"] for x in rows if x["test"]]
    out = {
        "cadence": cd.__version__,
        "brain": f"{a.brain.parent.name}/{a.brain.name}",
        "brain_cfg": cfg.get("name"),
        "parameters": int(brain.parameters()),
        "made": time.strftime("%Y-%m-%d"),
        "mode": "greedy, cold settles, reset + fixed warm-up per prompt, no learning",
        "warmup": warm,
        "vocab": [w for w in W.VOCAB if w not in (W.EOL, "you:", "bot:")],
        "counts": {"prompts": len(rows), "valid": sum(x["valid"] for x in rows),
                   "test_prompts": len(tv), "test_valid": sum(tv)},
        "prompts": rows,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=0), encoding="utf-8", newline="\n")
    print(json.dumps(out["counts"]), f"{time.time() - t0:.0f}s ->", a.out)


if __name__ == "__main__":
    main()
