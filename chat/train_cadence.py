"""Train one Cadence brain on the chat corpus, one character per moment.

    python train_cadence.py --cfg hist --chars 100000 --probe 25000 --out runs/screen/hist
    python train_cadence.py --cfg hist --chars 500000 --resume runs/x/brain.npz --start 100000 --out runs/x

Logs one JSON line per probe to <out>/log.jsonl and the full evaluation to eval-<chars>.json.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

for k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")

import textbrain as tb  # noqa: E402

H3 = {"history_decays": (0.2, 0.8, 0.95)}
CONFIGS = {
    # the founder composition, one char in: context only from the built-in working trace
    "plain": {"window": 1, "genes": {"learning_surprise": True}},
    # + heard history: fading copies of the input at three timescales
    "hist": {"window": 1, "genes": {"learning_surprise": True, **H3}},
    # the last 3 chars pressed into one observation, + heard history
    "win3": {"window": 3, "genes": {"learning_surprise": True, **H3}},
    "win3_m256": {"window": 3, "genes": {"learning_surprise": True, "modules": (256,), **H3}},
    "win3_norm": {"window": 3, "genes": {"learning_surprise": True, "teaching": "normalized", **H3}},
    "win3_m256_norm": {"window": 3, "genes": {"learning_surprise": True, "teaching": "normalized",
                                              "modules": (256,), **H3}},
    "hist_m256": {"window": 1, "genes": {"learning_surprise": True, "modules": (256,), **H3}},
    "win5_m256": {"window": 5, "genes": {"learning_surprise": True, "modules": (256,), **H3}},
    "win5_m256_norm": {"window": 5, "genes": {"learning_surprise": True, "teaching": "normalized",
                                              "modules": (256,), **H3}},
    "win8_m256_norm": {"window": 8, "genes": {"learning_surprise": True, "teaching": "normalized",
                                              "modules": (256,), **H3}},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--chars", type=int, default=100_000)
    ap.add_argument("--probe", type=int, default=25_000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--start", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    log_path = a.out / "log.jsonl"

    def log(line: str) -> None:
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")

    cfg = dict(CONFIGS[a.cfg], name=a.cfg)
    tb.train(cfg, a.out, a.chars, a.probe, seed=a.seed, resume=a.resume, start_char=a.start, log=log)


if __name__ == "__main__":
    main()
