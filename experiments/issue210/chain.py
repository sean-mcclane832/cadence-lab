"""Issue #210 checks, part 4: the prerequisite chain with memory_choice=True under each rule.

Declared before any run (2026-10-27): the upstream chamber
`benchmarks/chain/prerequisite_chain.py` (four steps in order, two idle actions, one
continuing `live` life of 20,000 moments, the `gene` arm at the protocol's founder point)
run through its `run_life`, except that `Brain.compose` is called with `memory_choice=True`
and without the chamber's `memory_rate=1.0` override (the episodic founder 0.5 applies, as
in the composition the owner's #210 panel reads). Modes `observed` (the stage is in the
observation) and `hidden` (the same observation at every stage, the hidden-chain case where
a paid action meets zero outcomes at other stages: the risk for any extinction rule), pays
`end` and `shaped`. Readings: completions per 1,000 moments lived, and greedy completions
per 1,000 of a saved copy at the end. Rules from rules.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
UPSTREAM = Path("P:/projects/cadence/benchmarks/chain")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(UPSTREAM))


def life(seed: int, mode: str, pay: str, rule: str, theta: float) -> dict:
    import warnings

    warnings.simplefilter("ignore")
    import cadence as cd
    import prerequisite_chain as pc

    import rules

    rules.apply(rule, theta)
    compose = rules.compose()

    def composed(inputs, actions, **kw):
        kw.pop("memory_rate", None)
        return compose(inputs, actions, memory_choice=True, **kw)

    cd.Brain.compose = composed  # type: ignore[assignment]
    protocol = json.loads((UPSTREAM / "protocol.json").read_text(encoding="utf-8"))
    try:
        with tempfile.TemporaryDirectory() as d:
            result = pc.run_life("gene", mode, pay, seed, protocol, Path(d))
    finally:
        cd.Brain.compose = rules.RELEASED_COMPOSE  # type: ignore[assignment]
    out = {"seed": seed, "mode": mode, "pay": pay, "rule": rule, "theta": theta,
           "writes": rules.STATE["writes"], "zeros": rules.STATE["zeros"],
           "lived": result["totals"]["completions_per_1000"],
           "greedy_end": result["final"]["greedy"]["completions_per_1000"],
           "sampled_end": result["final"]["sampled"]["completions_per_1000"],
           "seconds": round(result["seconds"], 1)}
    rules.apply("off")
    return out


def _job(kwargs: dict) -> dict:
    try:
        return life(**kwargs)
    except Exception as error:  # noqa: BLE001
        return {"error": f"{type(error).__name__}: {error}"[:300], **kwargs}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(6)))
    p.add_argument("--modes", nargs="+", default=["observed", "hidden"])
    p.add_argument("--pays", nargs="+", default=["end", "shaped"])
    p.add_argument("--rules", nargs="+", default=["off", "owner", "direct"])
    p.add_argument("--theta", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    jobs = [dict(seed=s, mode=m, pay=py, rule=r, theta=a.theta)
            for m in a.modes for py in a.pays for r in a.rules for s in a.seeds]
    t0 = time.time()
    with get_context("spawn").Pool(a.workers) as pool:
        rows = pool.map(_job, jobs, chunksize=1)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"rows": rows, "seconds": round(time.time() - t0, 1)}, indent=1),
                     encoding="utf-8", newline="\n")
    print(f"{len(rows)} lives, {time.time() - t0:.0f}s, errors: {sum('error' in r for r in rows)}")
    groups = defaultdict(list)
    for r in rows:
        if "error" in r:
            print("ERROR", r)
            continue
        groups[(r["mode"], r["pay"], r["rule"])].append(r)
    print("| mode | pay | rule | completions / 1000 lived | greedy at end (seeds > 0) | sampled at end | zero writes / zeros |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for (m, py, rule), rs in sorted(groups.items()):
        print(f"| {m} | {py} | {rule} | {np.mean([r['lived'] for r in rs]):.1f} |"
              f" {np.mean([r['greedy_end'] for r in rs]):.1f} ({sum(r['greedy_end'] > 0 for r in rs)}) |"
              f" {np.mean([r['sampled_end'] for r in rs]):.1f} | {sum(r['writes'] for r in rs)} / {sum(r['zeros'] for r in rs)} |")


if __name__ == "__main__":
    main()
