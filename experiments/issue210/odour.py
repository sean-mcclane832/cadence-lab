"""Issue #210 checks, part 3: the odour nursery with memory_choice=True under each rule.

Declared before any run (2026-10-27): the upstream chamber
`benchmarks/reversal/odour_nursery.py` (protocol odour-nursery/3, A -> B -> A, 600 trials
after each change, wrong approaches cost 1, avoiding pays 0) run unchanged through its
`run_life` for the `live` and `step` arms at exposures 300 and 1000, except that
`Brain.compose` is called with `memory_choice=True`. Readings as the owner's: lives passing
every check (live: acquired, reversed, returned, stable kept, calm; step: the four learning
checks, `step` is never calm), and the median reversal lag. Rules from rules.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
UPSTREAM = Path("P:/projects/cadence/benchmarks/reversal")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(UPSTREAM))


def life(seed: int, exposure: int, arm: str, rule: str, theta: float) -> dict:
    import warnings

    warnings.simplefilter("ignore")
    import cadence as cd
    import odour_nursery as on

    import rules

    rules.apply(rule, theta)
    compose = rules.compose()

    def composed(inputs, actions, **kw):
        return compose(inputs, actions, memory_choice=True, **kw)

    cd.Brain.compose = composed  # type: ignore[assignment]
    protocol = json.loads((UPSTREAM / "protocol.json").read_text(encoding="utf-8"))
    try:
        result = on.run_life(arm, seed, exposure, protocol)
    finally:
        cd.Brain.compose = rules.RELEASED_COMPOSE  # type: ignore[assignment]
    g = protocol["gates"]
    out = {"seed": seed, "exposure": exposure, "arm": arm, "rule": rule, "theta": theta,
           "writes": rules.STATE["writes"], "zeros": rules.STATE["zeros"]}
    rules.apply("off")
    if "error" in result:
        out["error"] = result["error"]
        return out
    ph = result["phases"]
    checks = {
        "acquired": ph[0]["final"] >= g["final"],
        "reversed": ph[1]["final"] >= g["final"],
        "returned": ph[2]["final"] >= g["final"],
        "stable_kept": all((p["stable"] or 0.0) >= g["stable"] for p in ph),
    }
    if arm == "live":
        checks["calm"] = all(p["aroused_late"] <= g["aroused_late"] for p in ph)
    out["checks"] = checks
    out["passed"] = all(checks.values())
    out["lag"] = ph[1]["lag"]
    out["finals"] = [p["final"] for p in ph]
    return out


def _job(kwargs: dict) -> dict:
    try:
        return life(**kwargs)
    except Exception as error:  # noqa: BLE001
        return {"error": f"{type(error).__name__}: {error}"[:300], **kwargs}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    p.add_argument("--exposures", type=int, nargs="+", default=[300, 1000])
    p.add_argument("--arms", nargs="+", default=["live", "step"])
    p.add_argument("--rules", nargs="+", default=["off", "owner", "direct"])
    p.add_argument("--theta", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    jobs = [dict(seed=s, exposure=e, arm=arm, rule=r, theta=a.theta)
            for arm in a.arms for e in a.exposures for r in a.rules for s in a.seeds]
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
        groups[(r["arm"], r["exposure"], r["rule"])].append(r)
    print("| arm | exposure | rule | lives passing | median reversal lag | zero writes / zeros |")
    print("| --- | --- | --- | --- | --- | --- |")
    for (arm, e, rule), rs in sorted(groups.items()):
        lags = [np.inf if r["lag"] is None else r["lag"] for r in rs]
        lag = np.median(lags)
        print(f"| {arm} | {e} | {rule} | {sum(r['passed'] for r in rs)} of {len(rs)} |"
              f" {'never' if np.isinf(lag) else f'{lag:.1f}'} | {sum(r['writes'] for r in rs)} / {sum(r['zeros'] for r in rs)} |")


if __name__ == "__main__":
    main()
