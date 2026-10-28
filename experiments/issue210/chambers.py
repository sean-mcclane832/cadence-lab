"""Issue #210 checks, part 1: the issue's bandit (A -> B -> A) and a songs chamber.

Declared before any run (2026-10-27):

Bandit (the issue's reproducer, extended with a return): one constant observation, two arms,
arm 0 pays for 120 trials, arm 1 for 120, arm 0 again for 120; the unpaid arm pays FAIL
(0 or -1). Brain.compose(1, 2, modules=(16,), arousal=True, memory_choice=True) through
`live`. Readings: B acquired = hit rate over the last 40 trials of B >= 0.8 (the issue's
criterion); return acquired = the same over the last 40 trials of the return; trials to
acquire B / return = first trial after the flip where the trailing 20 reach 0.9.
With --cue the observation is [1, last arm one-hot (2), last outcome > 0].

Songs (a reconstruction; the owner's songs chamber is not in the repository): two songs of
four notes over four actions, A = 0 1 2 3, B = 3 2 1 0. A song is cued once: its first
observation names the song, the next three are the same "continue" input. Ten lessons; a
lesson teaches both songs through `step(obs, teacher=[note])` with no reward (zero outcomes
mid-attempt), `done` after each song. Playback: `reset()`, then greedy `act` through the cue
and three continues, per song; score = share of the 8 notes right. Control:
memory_choice=False. Gate (as the owner's): pass if score >= control + 0.10. With --cue the
observation also carries the last note one-hot (4) and whether it was a demonstration match
(always 0 here, since songs pay nothing), so the cue there is "last note".

Rules: see rules.py (off, owner, direct), theta 0.5 unless given.
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
sys.path.insert(0, str(HERE))

PHASE = 120


def bandit_life(seed: int, rule: str, theta: float, fail: float, cue: bool, phases: int = 3) -> dict:
    import warnings

    warnings.simplefilter("ignore")
    import cadence as cd

    import rules

    rules.apply(rule, theta)
    width = 4 if cue else 1
    brain = rules.compose()(width, 2, modules=(16,), seed=seed, arousal=True, memory_choice=True)

    def obs(last: int | None, won: bool) -> np.ndarray:
        x = np.zeros((1, width))
        x[0, 0] = 1.0
        if cue and last is not None:
            x[0, 1 + last] = 1.0
            x[0, 3] = float(won)
        return x

    action = int(brain.live(obs(None, False))[0])
    hits = []
    for t in range(phases * PHASE):
        target = (t // PHASE) % 2
        reward = 1.0 if action == target else fail
        hits.append(action == target)
        action_next = int(brain.live(obs(action, reward > 0), reward=[reward])[0])
        action = action_next
    h = np.asarray(hits, float)
    out = {"seed": seed, "rule": rule, "theta": theta, "fail": fail, "cue": cue,
           "writes": rules.STATE["writes"], "zeros": rules.STATE["zeros"]}
    for p in range(phases):
        seg = h[p * PHASE:(p + 1) * PHASE]
        out[f"late{p}"] = float(seg[-40:].mean())
        trail = np.convolve(seg, np.ones(20) / 20, mode="valid")
        hit = np.flatnonzero(trail >= 0.9)
        out[f"lag{p}"] = int(hit[0] + 20) if len(hit) else None
    rules.apply("off")
    return out


SONGS = {0: [0, 1, 2, 3], 1: [3, 2, 1, 0]}


def song_life(seed: int, rule: str, theta: float, episodic: bool, cue: bool, lessons: int = 10) -> dict:
    import warnings

    warnings.simplefilter("ignore")
    import cadence as cd

    import rules

    rules.apply(rule, theta)
    width = 3 + (4 if cue else 0)
    brain = rules.compose()(width, 4, modules=(16,), seed=seed, memory_choice=episodic)

    def obs(position: int, song: int, last: int | None) -> np.ndarray:
        x = np.zeros((1, width))
        x[0, song if position == 0 else 2] = 1.0
        if cue and last is not None:
            x[0, 3 + last] = 1.0
        return x

    first = True
    for _ in range(lessons):
        for song, notes in SONGS.items():
            last = None
            for i, note in enumerate(notes):
                feedback = {}
                if not first:
                    feedback = {"reward": np.zeros(1), "done": np.array([i == 0])}
                brain.step(obs(i, song, last), teacher=np.array([note]), **feedback)
                first = False
                last = note
    # close the last attempt
    brain.step(obs(0, 0, None), reward=np.zeros(1), done=np.array([True]))
    right = 0
    for song, notes in SONGS.items():
        brain.reset()
        last = None
        for i, note in enumerate(notes):
            a = int(brain.act(obs(i, song, last), greedy=True)[0])
            right += a == note
            last = a
    out = {"seed": seed, "rule": rule, "theta": theta, "episodic": episodic, "cue": cue,
           "score": right / 8.0, "writes": rules.STATE["writes"], "zeros": rules.STATE["zeros"]}
    rules.apply("off")
    return out


def _job(spec: tuple) -> dict:
    kind, kwargs = spec
    try:
        return (bandit_life if kind == "bandit" else song_life)(**kwargs)
    except Exception as error:  # noqa: BLE001
        return {"error": f"{type(error).__name__}: {error}"[:300], **kwargs, "kind": kind}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(12)))
    p.add_argument("--rules", nargs="+", default=["off", "owner", "direct"])
    p.add_argument("--theta", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--only", choices=["bandit", "songs"], default=None)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    jobs = []
    if a.only in (None, "bandit"):
        for rule in a.rules:
            for fail in (0.0, -1.0):
                for cue in (False, True):
                    if cue and rule != "off":
                        continue
                    for s in a.seeds:
                        jobs.append(("bandit", dict(seed=s, rule=rule, theta=a.theta, fail=fail, cue=cue)))
    if a.only in (None, "songs"):
        for s in a.seeds:
            jobs.append(("songs", dict(seed=s, rule="off", theta=a.theta, episodic=False, cue=False)))
            jobs.append(("songs", dict(seed=s, rule="off", theta=a.theta, episodic=True, cue=True)))
            for rule in a.rules:
                jobs.append(("songs", dict(seed=s, rule=rule, theta=a.theta, episodic=True, cue=False)))
    t0 = time.time()
    with get_context("spawn").Pool(a.workers) as pool:
        rows = pool.map(_job, jobs, chunksize=1)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"rows": rows, "seconds": round(time.time() - t0, 1)}, indent=1), encoding="utf-8", newline="\n")
    print(f"{len(rows)} lives, {time.time() - t0:.0f}s, errors: {sum('error' in r for r in rows)}")


if __name__ == "__main__":
    main()
