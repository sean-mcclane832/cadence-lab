"""Serial reversal recheck: is the 10/27 learning-set curve learning-to-learn, or mechanics?

Run:  python experiments/serial_reversal_recheck.py --part controls --seeds 0 1 2 3 4 5 6 7 8 9 10 11
      python experiments/serial_reversal_recheck.py --part lesions  --seeds ...
      python experiments/serial_reversal_recheck.py --part wm0      --seeds ...

Protocol and predictions: notes/serial_reversal_recheck.md (committed before any run).

controls: every condition ends in the same test: a 100-moment block where arm 0 pays, then a flip
  to arm 1; errors = wrong-arm choices in the first 20 moments after the flip (as in v1).
  naive      only the 100-moment block.
  stable     1100 moments of arm 0 paying first (age-matched to the serial life, no flip).
  serial     the v1 schedule (first 200, then 9 jittered flips), then the test block and flip;
             the test is reversal 11 of its life (arm 0 pays in the test block after an even
             number of flips, so 10 flips happen before the test block; see `segments`).
  both_pay   1000 moments where both arms pay +1 (no losses, no flips), then the test.
  coin       1000 moments where the paying arm is drawn per moment (50/50), then the test.
  Arms: continuing, memory_choice (the v1 brains). Also pre-flip errors: wrong choices in the
  last 20 moments before every flip (stability).

lesions: the serial brain (and a naive one) is copied at the moment of the test flip and the 20
  test moments are replayed under: intact, store cleared (hippocampus.clear: fast and
  consolidated records), consolidated records only zeroed, working-memory trace zeroed for the
  window (amplitude 0), arousal reset (brain.arousal.reset()), arousal_swap (the other brain's
  arousal state: serial gets the naive brain's, naive gets the serial one's), store_swap (the
  other brain's hippocampus records, fast and consolidated).

wm0: the v1 serial run with working_memory_amplitude=0 from birth (the built-in copy silenced).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import cadence as cd
from cadence import Brain

from envs.serial import SerialBandit
from bandit_reversal import live

WINDOW = 20
OBS = np.ones((1, 1))


def make_brain(arm: str, seed: int, **extra) -> Brain:
    kw = dict(extra)
    if arm == "memory_choice":
        kw["memory_choice"] = True
    return Brain.compose(1, 2, modules=(16,), seed=seed, arousal=True, **kw)


def segments(cond: str, seed: int) -> list[tuple[str, int, int]]:
    """(kind, length, paying arm) segments; kind 'arm' = one arm pays, 'both', 'coin'."""
    test = [("arm", 100, 0), ("arm", WINDOW, 1)]
    if cond == "naive":
        return test
    if cond == "stable":
        return [("arm", 1100, 0)] + test[1:]  # 1100 moments of arm 0, then the flip
    if cond == "both_pay":
        return [("both", 1000, -1)] + test
    if cond == "coin":
        return [("coin", 1000, -1)] + test
    if cond == "serial":
        env = SerialBandit(reversals=10, seed=seed)  # v1 schedule, 10 flips
        lengths = np.diff([0] + env.flips + [env.life])
        segs = [("arm", int(n), k % 2) for k, n in enumerate(lengths)]
        # 11 blocks alternate 0,1,...,0 ; the last v1 block (arm 0) is replaced by the test block
        segs[-1] = ("arm", 100, 0)
        return segs + [("arm", WINDOW, 1)]
    raise ValueError(cond)


def run(arm: str, cond: str, seed: int, *, snapshot: bool = False) -> dict:
    segs = segments(cond, seed)
    brain = make_brain(arm, seed)
    rng = np.random.default_rng(seed + 555)
    hits, flips, t = [], [], 0
    snap = None
    action = live(brain, OBS)
    for i, (kind, n, pay) in enumerate(segs):
        if i > 0 and kind == "arm" and segs[i - 1][0] == "arm" and segs[i - 1][2] != pay:
            flips.append(t)
        if snapshot and i == len(segs) - 1:
            snap = (copy.deepcopy(brain), copy.deepcopy(action))
        for _ in range(n):
            a = int(action[0])
            if kind == "both":
                r, hit = 1.0, True
            elif kind == "coin":
                hit = a == int(rng.integers(2))
                r = 1.0 if hit else -1.0
            else:
                hit = a == pay
                r = 1.0 if hit else -1.0
            hits.append(hit)
            action = live(brain, OBS, r)
            t += 1
    h = np.asarray(hits, dtype=float)
    test = flips[-1]
    out = {"arm": arm, "cond": cond, "seed": seed, "flips": flips, "life": t,
           "test_errors": int(WINDOW - h[test:test + WINDOW].sum()),
           "pre_flip_errors": [int(WINDOW - h[f - WINDOW:f].sum()) for f in flips],
           "post_flip_errors": [int(WINDOW - h[f:f + WINDOW].sum()) for f in flips]}
    return out, snap


def replay(brain: Brain, action, lesion: str, other: Brain | None = None) -> int:
    b = copy.deepcopy(brain)
    a0 = copy.deepcopy(action)
    if lesion == "arousal_swap":
        b.arousal = copy.deepcopy(other.arousal)
    elif lesion == "store_swap":
        b.hippocampus.consolidated[...] = other.hippocampus.consolidated
        b.hippocampus.strength[...] = other.hippocampus.strength
    if lesion == "store_clear" and b.hippocampus is not None:
        b.hippocampus.clear()
    elif lesion == "consolidated_zero" and b.hippocampus is not None:
        b.hippocampus.consolidated.fill(0)
    elif lesion == "arousal_reset" and b.arousal is not None:
        b.arousal.reset()
    elif lesion == "wm0" and b.working_memory is not None:
        b.working_memory.amplitude = 0.0
    errors = 0
    action = a0
    for _ in range(WINDOW):
        a = int(action[0])
        hit = a == 1
        errors += int(not hit)
        action = live(b, OBS, 1.0 if hit else -1.0)
    return errors


LESIONS = ("intact", "store_clear", "consolidated_zero", "wm0", "arousal_reset", "arousal_swap", "store_swap")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("controls", "lesions", "wm0"), required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(12)))
    ap.add_argument("--arms", nargs="+", default=["continuing", "memory_choice"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    outdir = ROOT / "runs" / "serial_reversal_recheck"
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []

    if args.part == "controls":
        for arm in args.arms:
            for cond in ("naive", "stable", "serial", "both_pay", "coin"):
                for seed in args.seeds:
                    t0 = time.perf_counter()
                    row, _ = run(arm, cond, seed)
                    row["elapsed_s"] = round(time.perf_counter() - t0, 2)
                    rows.append(row)
                    print(f"{arm:13s} {cond:9s} seed {seed:2d}: test errors {row['test_errors']:2d}  "
                          f"pre-flip {row['pre_flip_errors']}  post {row['post_flip_errors']}", flush=True)
        print()
        for arm in args.arms:
            for cond in ("naive", "stable", "serial", "both_pay", "coin"):
                e = [r["test_errors"] for r in rows if r["arm"] == arm and r["cond"] == cond]
                print(f"SUMMARY {arm:13s} {cond:9s} test errors mean {np.mean(e):5.2f} "
                      f"median {np.median(e):4.1f}  {e}")

    elif args.part == "lesions":
        for arm in args.arms:
            for seed in args.seeds:
                snaps = {c: run(arm, c, seed, snapshot=True) for c in ("serial", "naive")}
                for cond in ("serial", "naive"):
                    row, (brain, action) = snaps[cond]
                    other = snaps["naive" if cond == "serial" else "serial"][1][0]
                    res = {les: replay(brain, action, les, other) for les in LESIONS}
                    rows.append({"arm": arm, "cond": cond, "seed": seed, "unreplayed": row["test_errors"],
                                 **res})
                    print(f"{arm:13s} {cond:7s} seed {seed:2d}: " + " ".join(f"{k} {v:2d}" for k, v in res.items())
                          + f" (run {row['test_errors']})", flush=True)
        print()
        for arm in args.arms:
            for cond in ("serial", "naive"):
                rs = [r for r in rows if r["arm"] == arm and r["cond"] == cond]
                print(f"SUMMARY {arm:13s} {cond:7s} " +
                      " ".join(f"{k} {np.mean([r[k] for r in rs]):5.2f}" for k in LESIONS))

    else:  # wm0 from birth, v1 serial schedule
        sys.argv = [sys.argv[0]]
        import serial_reversal as sr
        orig = sr.make_brain
        sr.make_brain = lambda arm, seed: Brain.compose(
            1, 2, modules=(16,), seed=seed, arousal=True, episodic=(arm != "memory_off"),
            working_memory_amplitude=0.0, **({"memory_choice": True} if arm == "memory_choice" else {}))
        for arm in args.arms:
            for seed in args.seeds:
                env = SerialBandit(10, 200, 100, 20, 1.0, seed=seed)
                r = sr.run_brain(arm, env, seed)
                revs = sr.per_reversal(r["hits"], env)
                errs = [v["errors"] for v in revs]
                rows.append({"arm": arm, "seed": seed, "errors": errs, "slope": sr.slope(errs)})
                print(f"{arm:13s} wm0 seed {seed:2d}: errors {errs} slope {sr.slope(errs):+.2f}", flush=True)
        sr.make_brain = orig
        print()
        for arm in args.arms:
            rs = [r for r in rows if r["arm"] == arm]
            e = np.array([r["errors"] for r in rs], dtype=float)
            print(f"SUMMARY {arm:13s} wm0 first3 {e[:, :3].mean():5.2f} last3 {e[:, -3:].mean():5.2f} "
                  f"slope {np.mean([r['slope'] for r in rs]):+.3f} neg {sum(r['slope'] < 0 for r in rs)}/{len(rs)}")

    body = {"part": args.part, "seeds": args.seeds, "arms": args.arms, "window": WINDOW,
            "versions": {"cadence": getattr(cd, "__version__", "unknown"), "numpy": np.__version__,
                         "python": sys.version.split()[0]},
            "rows": rows}
    sources = [("experiments/serial_reversal_recheck.py", Path(__file__)),
               ("envs/serial.py", ROOT / "envs" / "serial.py"),
               ("experiments/serial_reversal.py", ROOT / "experiments" / "serial_reversal.py")]
    receipt = cd.Receipt.build(f"cadence-lab/serial-reversal-recheck/{args.part}", body, sources=sources)
    path = receipt.write(Path(args.out) if args.out else outdir / f"{args.part}.json")
    ok, msg = cd.Receipt.verify(path, sources=sources)
    print(f"\nreceipt: {path} ({msg})")


if __name__ == "__main__":
    main()
