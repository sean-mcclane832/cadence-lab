"""Same composed brain, same life, on CPU (NumPy+Numba float64) vs torch CUDA float64 / float32.

Run:  python experiments/backend_timing.py --sizes 64 512 --moments 300
Task: 4 one-hot cues, 2 actions, the paying action is cue % 2 (+1 / -1).
A brain is composed on the CPU, saved, and loaded on each backend so every arm
starts from identical parameters. Writes runs/backend_timing/receipt.json.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import cadence as cd
import numpy as np
import torch
from cadence import Brain

ROOT = Path(__file__).resolve().parents[1]
ARMS = [("cpu", {"backend": "cpu"}),
        ("cuda-float64", {"backend": "torch", "precision": "float64"}),
        ("cuda-float32", {"backend": "torch", "precision": "float32"})]


def life(brain: Brain, moments: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    cues = np.eye(4)
    cue = int(rng.integers(4))
    action = brain.live(cues[[cue]])
    hits, refusals = [], 0
    t0 = time.perf_counter()
    for _ in range(moments):
        hit = int(action[0]) == cue % 2
        hits.append(hit)
        cue = int(rng.integers(4))
        try:
            action = brain.live(cues[[cue]], reward=[1.0 if hit else -1.0])
        except RuntimeError:
            refusals += 1
            action = brain.live(cues[[cue]])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    a = brain.arousal
    return {"ms_per_moment": 1000 * elapsed / moments, "elapsed_s": elapsed,
            "hit_last100": float(np.mean(hits[-100:])), "refusals": refusals,
            "answer_sweeps": int(sum(a.sweeps.values())), "learning_sweeps": int(a.learning_sweeps),
            "aroused_moments": int(a.moments.get("aroused", 0))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[64, 512])
    ap.add_argument("--moments", type=int, default=300)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--out", default=str(ROOT / "runs" / "backend_timing" / "receipt.json"))
    args = ap.parse_args()
    rows = []
    tmp = Path(tempfile.mkdtemp())
    for size in args.sizes:
        for seed in args.seeds:
            path = tmp / f"b{size}-{seed}.npz"
            Brain.compose(4, 2, modules=(size,), seed=seed, arousal=True).save(path)
            for name, kw in ARMS:
                brain = Brain.load(path, **kw)
                brain.live(np.eye(4)[[0]])  # warm-up moment (JIT / CUDA init), not timed
                brain = Brain.load(path, **kw)
                r = {"size": size, "seed": seed, "arm": name, **life(brain, args.moments, seed)}
                rows.append(r)
                print(f"modules=({size},) seed {seed} {name:13s} {r['ms_per_moment']:7.2f} ms/moment "
                      f"hit_last100 {r['hit_last100']:.2f} sweeps {r['answer_sweeps'] + r['learning_sweeps']}",
                      flush=True)
    body = {"protocol": vars(args) | {"task": "4 one-hot cues, 2 actions, pay cue%2 (+1/-1)"},
            "versions": {"cadence": "0.89.0", "torch": torch.__version__, "python": sys.version.split()[0],
                         "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                         "backends": cd.available_backends()},
            "rows": rows,
            "boundary": "Wall-clock on one desktop (i7-9700K, RTX 2060 SUPER). Small brains; not a throughput benchmark."}
    receipt = cd.Receipt.build("cadence-lab/backend-timing/v1", body,
                               sources=[("experiments/backend_timing.py", Path(__file__))])
    print("receipt:", receipt.write(Path(args.out)))


if __name__ == "__main__":
    main()
