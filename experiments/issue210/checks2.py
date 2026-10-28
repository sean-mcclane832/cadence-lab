"""Issue #210 checks, part 5: the owner's remaining preservation checks that live in the repository.

Declared before any run (2026-10-28, overnight): three chambers, each with `memory_choice=True`,
under the rules off (0.89.0), owner (his gene as described on #210, rebuilt in rules.py) and
lib (the direct-payment gene of P:/projects/cadence-210fix, `memory_extinction=theta`).

delayed   tests/test_context_defaults.py's delayed response from reward: a cue, then a go
          signal with the cue gone, reward only at go, batch 32, 1500 episodes, `act`/`learn`.
          Reading: greedy hit rate over 5 fresh batches (chance 0.5; the test asks >= 0.9).
          The step after the cue is a zero outcome mid-attempt; the cue-step action is never
          paid directly, so the prediction is that lib writes nothing and is identical to off.
          The owner reported "identical lives" for his gene here.
bandit4   tests/test_context_defaults.py's continuing contextual bandit: four one-hot
          contexts, the matching action pays 1, the others 0, 600 decisions through `step`,
          `actor_gamma=0, actor_lam=0`, greedy hit rate of a frozen copy on 200 contexts
          (chance 0.25). A wrong action is never paid in its context, so lib should write
          nothing unless overlapping keys leak a payment flag (the 0.9 flag guards this).
          The owner reported his gene unchanged at .5.
taughtcue tests/test_episodic_memory.py's taught cue recall, scaled up: one demonstration of
          each cue's answer per lesson (cue step taught, go step taught with reward 0, which is
          a learned zero right after a demonstration), `lessons` lessons, then greedy recall
          at go after each cue with the cue gone. Reading: share right over both cues
          (chance 0.5). A taught act is never a direct payment, so lib should write nothing.

Not run, because they are not in the repository (the owner's own chambers): the crafting
chains taught and alone, V5 corrective and reward only, and the original songs chamber.

Seeds: delayed 8-19 (the test's 8-13 plus six more), bandit4 0-23, taughtcue 0-23.
Prediction overall: lib identical to off on every life, 0 writes.
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
sys.path.insert(0, str(HERE))


def delayed(seed: int, rule: str, theta: float, episodes: int = 1500) -> dict:
    import rules

    rules.apply(rule, theta)
    brain = rules.compose()(3, 2, seed=seed, memory_choice=True)
    rng = np.random.default_rng(seed)
    batch = 32

    def episode():
        cue = rng.integers(0, 2, size=batch)
        shown, go = np.zeros((batch, 3)), np.zeros((batch, 3))
        shown[np.arange(batch), cue] = 1.0
        go[:, 2] = 1.0
        return cue, shown, go

    for _ in range(episodes):
        cue, shown, go = episode()
        brain.reset()
        brain.act(shown)
        brain.learn(np.zeros(batch), np.zeros(batch, dtype=bool), go)
        action = brain.act(go)
        brain.learn((action == cue).astype(float), np.ones(batch, dtype=bool), go)
    hits = []
    for _ in range(5):
        cue, shown, go = episode()
        brain.reset()
        brain.act(shown, greedy=True)
        hits.append(float((brain.act(go, greedy=True) == cue).mean()))
    return {"score": float(np.mean(hits))}


def bandit4(seed: int, rule: str, theta: float, decisions: int = 600) -> dict:
    import rules

    rules.apply(rule, theta)
    brain = rules.compose()(4, 4, seed=seed, memory_choice=True, actor_gamma=0.0, actor_lam=0.0)
    rng = np.random.default_rng(seed)
    context = int(rng.integers(0, 4))
    action = brain.step(np.eye(4)[[context]])
    for _ in range(decisions - 1):
        reward = float(action[0] == context)
        context = int(rng.integers(0, 4))
        action = brain.step(np.eye(4)[[context]], reward=[reward])
    hits = 0
    for _ in range(200):
        context = int(rng.integers(0, 4))
        hits += int(brain.act(np.eye(4)[[context]], greedy=True)[0] == context)
    return {"score": hits / 200}


def taughtcue(seed: int, rule: str, theta: float, lessons: int = 3) -> dict:
    import rules

    rules.apply(rule, theta)
    brain = rules.compose()(3, 2, seed=seed, memory_choice=True)
    eye = np.eye(3)
    cues = {0: eye[[0]], 1: eye[[1]]}
    go = eye[[2]]
    for _ in range(lessons):
        for cue, shown in cues.items():
            brain.reset()
            brain.step(shown, teacher=[cue])
            brain.step(go, reward=[0.0], teacher=[cue])
    right = 0
    for cue, shown in cues.items():
        brain.reset()
        brain.act(shown, greedy=True)
        right += int(brain.act(go, greedy=True)[0]) == cue
    return {"score": right / 2}


TASKS = {"delayed": delayed, "bandit4": bandit4, "taughtcue": taughtcue}


def _job(spec: dict) -> dict:
    import warnings

    warnings.simplefilter("ignore")
    import rules

    try:
        out = TASKS[spec["task"]](spec["seed"], spec["rule"], spec["theta"])
        out.update(spec, writes=rules.STATE["writes"], zeros=rules.STATE["zeros"])
    except Exception as error:  # noqa: BLE001
        out = {**spec, "error": f"{type(error).__name__}: {error}"[:300]}
    rules.apply("off")
    return out


SEEDS = {"delayed": range(8, 20), "bandit4": range(24), "taughtcue": range(24)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rules", nargs="+", default=["off", "owner", "lib"])
    p.add_argument("--theta", type=float, default=0.5)
    p.add_argument("--tasks", nargs="+", default=list(TASKS))
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    jobs = [dict(task=t, seed=s, rule=r, theta=a.theta) for t in a.tasks for r in a.rules for s in SEEDS[t]]
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
        groups[(r["task"], r["rule"])].append(r)
    base = {(r["task"], r["seed"]): r["score"] for r in rows if r.get("rule") == "off" and "score" in r}
    print("| task | rule | mean score | seeds | identical to off | zero writes / zeros |")
    print("| --- | --- | --- | --- | --- | --- |")
    for (t, rule), rs in sorted(groups.items()):
        same = sum(base.get((t, r["seed"])) == r["score"] for r in rs)
        print(f"| {t} | {rule} | {np.mean([r['score'] for r in rs]):.3f} | {len(rs)} | {same} of {len(rs)} |"
              f" {sum(r['writes'] for r in rs)} / {sum(r['zeros'] for r in rs)} |")


if __name__ == "__main__":
    main()
