"""Talk to a saved Cadence brain.

    python chat.py                       # the shipped brain, brains/chat/brain.npz
    python chat.py --brain runs/x/brain.npz --temperature 0.3

Type a line and press enter. Lowercase letters, spaces and . , ? ! ' only; other
characters are dropped. Ctrl+C or "quit" to stop. The brain does not learn while you
talk; it keeps hearing the conversation (its working trace and heard history carry on).
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

for k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

import cadence as cd  # noqa: E402
import corpus as C  # noqa: E402
import textbrain as tb  # noqa: E402

HERE = Path(__file__).resolve().parent


def clean(line: str) -> str:
    line = line.lower().replace("\n", " ").replace(":", " ")
    return " ".join("".join(c for c in line if c in C.INDEX).split())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", type=Path, default=HERE / "brains" / "chat" / "brain.npz")
    ap.add_argument("--temperature", type=float, default=None,
                    help="sample replies at this temperature (default: greedy)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--say", nargs="*", help="non-interactive: these prompts, in one conversation")
    a = ap.parse_args()
    cfg = json.loads((a.brain.parent / "cfg.json").read_text(encoding="utf-8"))
    brain = cd.Brain.load(a.brain)
    brain.reset()
    rng = np.random.default_rng(a.seed)
    hist: list[int] = []
    window = cfg["window"]
    print(f"cadence {cd.__version__}, brain {a.brain.name} ({cfg.get('name')}), "
          f"{brain.parameters()} parameters. Type 'quit' to stop.")

    def turn(line: str) -> None:
        p = clean(line)
        if not p:
            return
        r = tb.say(brain, hist, p, window, temperature=a.temperature, rng=rng)
        print(f"bot: {r}", flush=True)

    if a.say is not None:
        for p in a.say:
            print(f"you: {clean(p)}")
            turn(p)
        return
    while True:
        try:
            line = input("you: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line.strip().lower() in {"quit", "exit"}:
            break
        turn(line)


if __name__ == "__main__":
    main()
