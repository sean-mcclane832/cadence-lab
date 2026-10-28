"""Issue #210 checks, part 2: the key-door nursery with memory_choice=True under each rule.

Declared before any run (2026-10-27): the upstream chamber `benchmarks/keydoor/key_door.py`
(protocol-3: chest -> lamp -> chest, 500 trips per rule, cost 0.25, 15 cells) run unchanged
through its `run_life` for the `live` arm, except: (a) `Brain.compose` is called with
`memory_choice=True` (the owner's readings on #210 are for that brain), (b) the policy probes
run only at the start and the end of each rule (they read saved copies and never touch the
living brain), and (c) with --cue the observation also carries the last action one-hot (2)
and whether the last outcome was positive (1). Readings as the owner's: trips to acquire A,
adapt to B, return to A (the chamber's `lag`, first 20-trip window with 90% fed), and lives
that return within 50 trips. Rules from rules.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
UPSTREAM = Path("P:/projects/cadence/benchmarks/keydoor")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(UPSTREAM))


def life(seed: int, delay: int, rule: str, theta: float, cue: bool) -> dict:
    import functools
    import warnings

    warnings.simplefilter("ignore")
    import cadence as cd
    import key_door as kd

    import rules

    rules.apply(rule, theta)
    compose = rules.compose()
    last = {"action": None, "won": False}

    def composed(inputs, actions, **kw):
        return compose(inputs + (3 if cue else 0), actions, memory_choice=True, **kw)

    cd.Brain.compose = composed  # type: ignore[assignment]
    observe = kd.__dict__.setdefault("_released_observe", kd.observe)
    act = kd.BrainLife.__dict__.get("_released_act") or kd.BrainLife.act
    kd.BrainLife._released_act = act
    kd.observe, kd.BrainLife.act = observe, act
    if cue:

        def cued(kind, holding, pouch):
            x = observe(kind, holding, pouch)
            extra = np.zeros((1, 3))
            if last["action"] is not None:
                extra[0, last["action"]] = 1.0
                extra[0, 2] = float(last["won"])
            return np.concatenate([x, extra], axis=1)

        kd.observe = cued

        def acting(self, kind, holding, reward, done):
            if reward is not None:
                last["won"] = reward > 0
            if done:
                last["action"], last["won"] = None, False
            action, aroused = act(self, kind, holding, reward, done)
            last["action"] = action
            return action, aroused

        kd.BrainLife.act = acting
    protocol = json.loads((UPSTREAM / "protocol-3.json").read_text(encoding="utf-8"))
    protocol["probe_every"] = 10**9
    protocol.setdefault("rules", ["chest", "lamp", "chest"])
    result = kd.run_life("live", seed, delay, protocol)
    cd.Brain.compose = rules.RELEASED_COMPOSE  # type: ignore[assignment]
    kd.observe, kd.BrainLife.act = observe, act
    out = {"seed": seed, "delay": delay, "rule": rule, "theta": theta, "cue": cue,
           "writes": rules.STATE["writes"], "zeros": rules.STATE["zeros"],
           "seconds": result.get("seconds"), "rules": protocol["rules"]}
    if "error" in result:
        out["error"] = result["error"]
        return out
    out["lags"] = [ph["lag"] for ph in result["phases"]]
    out["fed"] = [ph["fed"] for ph in result["phases"]]
    rules.apply("off")
    return out


def _job(kwargs: dict) -> dict:
    try:
        return life(**kwargs)
    except Exception as error:  # noqa: BLE001
        return {"error": f"{type(error).__name__}: {error}"[:300], **kwargs}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    p.add_argument("--delays", type=int, nargs="+", default=[2, 5])
    p.add_argument("--rules", nargs="+", default=["off", "owner", "direct"])
    p.add_argument("--cue", action="store_true", help="also run the cue arm (rule off)")
    p.add_argument("--theta", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    jobs = [dict(seed=s, delay=d, rule=r, theta=a.theta, cue=False)
            for d in a.delays for r in a.rules for s in a.seeds]
    if a.cue:
        jobs += [dict(seed=s, delay=d, rule="off", theta=a.theta, cue=True) for d in a.delays for s in a.seeds]
    t0 = time.time()
    with get_context("spawn").Pool(a.workers) as pool:
        rows = pool.map(_job, jobs, chunksize=1)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"rows": rows, "seconds": round(time.time() - t0, 1)}, indent=1),
                     encoding="utf-8", newline="\n")
    print(f"{len(rows)} lives, {time.time() - t0:.0f}s, errors: {sum('error' in r for r in rows)}")
    from collections import defaultdict

    groups = defaultdict(list)
    for r in rows:
        if "error" in r:
            print("ERROR", r)
            continue
        groups[(r["delay"], r["rule"], r["cue"])].append(r)
    print("| delay | arm | acquire A | adapt B | return A | returns within 50 | zero writes / zeros |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for (d, rule, cue), rs in sorted(groups.items()):
        def med(i):
            v = [r["lags"][i] for r in rs if r["lags"][i] is not None]
            miss = sum(r["lags"][i] is None for r in rs)
            return (f"{np.median(v):.1f}" if v else "-") + (f" ({miss} never)" if miss else "")
        ret = sum(r["lags"][2] is not None and r["lags"][2] <= 50 for r in rs)
        name = f"{rule}" + (" + cue" if cue else "")
        print(f"| {d} | {name} | {med(0)} | {med(1)} | {med(2)} | {ret} of {len(rs)} |"
              f" {sum(r['writes'] for r in rs)} / {sum(r['zeros'] for r in rs)} |")


if __name__ == "__main__":
    main()
