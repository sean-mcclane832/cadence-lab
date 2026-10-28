"""A classic animal-learning battery for one continuing Cadence brain, against a plain MLP.

Run:  python experiments/animal_battery.py --seeds $(seq 0 23)
Writes runs/animal_battery/receipt.json and prints one table per effect.

Declared before any run (2026-10-28). Upstream (muellerberndt/cadence 0.89.0, docs, tests,
benchmarks, cadence-demos) was searched first: none of these effects is tested there. The
only near hit is the documented absence of automatic extinction after zero outcomes
(docs/memory.md, issue #210), which is why every unreinforced press here costs a little.

The task: an operant analog of the classic designs. Each trial is two moments of one
continuing life through `live`: a stimulus moment and an inter-trial moment (no stimulus).
Two actions: 0 withhold, 1 press. Observation (6): [bias 1, A, B, C, context X, context Y];
one context is always on. Outcomes for the moment's action, given at the next moment:
  press on a reinforced stimulus trial  +1
  press on any other moment            -COST (effort; unreinforced presses, the inter-trial
                                             moment, every extinction trial; COST 0.5, see below)
  withhold                              0
"Press rate" is the share of stimulus trials on which the brain pressed (its own behaviour:
greedy when calm, sampled when aroused), read over the first TEST trials of a test phase run
under extinction (a press costs COST), as in an animal test session.

Arms (same seeds, same streams):
  cadence     Brain.compose(6, 2, modules=(16,), arousal=True, seed=s), founder genes.
  cadence_mc  the same with memory_choice=True (the episodic vote in the choice).
  mlp         6-16-2 tanh MLP Q-learner, epsilon-greedy 0.1, one SGD step (lr 0.05) per moment
              toward the moment's outcome (gamma 0), as the lab's earlier baselines.
CPU float64, Cadence 0.89.0 as released (no #210 gene).

Effects, predictions (what animals show), metrics:
  PREE        Partial reinforcement extinction effect. Groups p=1.0 and p=0.5: TRAIN trials
              of stimulus A, a press reinforced with probability p; then EXT trials of A in
              extinction. Animals: the p=0.5 group keeps pressing longer. Metrics: press rate
              over the first 50 extinction trials; trials to extinction (first extinction
              trial after which the trailing 10 A trials have press rate <= 0.2; EXT if never).
              A TD learner predicts the opposite (lower learned value under p=0.5, so it falls
              below the effort cost sooner).
  blocking    Kamin blocking. Group block: PHASE trials of A reinforced, then PHASE trials of
              the compound AB reinforced. Group control: the same, but phase 1 is C reinforced
              (matched experience). Test: B alone, TEST trials in extinction. Animals: block
              presses less to B than control. Also read A alone after phase 2 (block group) to
              show the brain answers A.
  renewal     ABA renewal. Acquire A reinforced in context X (PHASE trials), extinguish A in
              context Y (PHASE trials), test A in X (group ABA); control AAA (acquire, extinguish
              and test all in X), and ABB (test in the extinction context Y). Animals: ABA
              presses more than AAA and ABB.
  recovery    Spontaneous recovery. Acquire A in X (PHASE), extinguish A in X (PHASE), then
              test A in X either at once (group now) or after WAIT moments of passing time
              (group later): `wait(blank)` moments while the last extinction press awaits its
              outcome, so only the working trace and heard history move (parameters, critic,
              associative memory stay; see `Brain.wait`). The MLP has no time; its later group
              is its now group by construction (a control). Animals: later presses more than now.
  latent      Latent inhibition. Group pre: PHASE trials of A with no reinforcement (press costs
              COST as anywhere), then A reinforced for PHASE trials. Group none: the same number of
              trials with no stimulus (inter-trial moments only), then A reinforced. Animals:
              pre acquires more slowly. Metrics: press rate over the first 50 acquisition trials;
              trials to acquisition (trailing 10 A trials >= 0.8).

Parameters: TRAIN 300, EXT 300, PHASE 200, TEST 10, WAIT 500. Seeds 0-23 (24 per cell).

Revision after a pilot, declared before the full run (2026-10-28 00:25). A pilot on seed 0
(every cell) and on seeds 0-3 (extinction speed only, no group comparison read) showed that at
a press cost of 0.1 a Cadence brain barely extinguishes: still 62% presses after 200
extinction trials, so renewal and spontaneous recovery could not be read (nothing had been
extinguished), and the trials to an extinction criterion ranged 86 to 600+. At 0.5 the four
seeds reached the criterion in 10-38 (cadence) and 17-257 (cadence_mc) trials, the MLP in
10-13. So, for the full run:
  - COST is 0.5 (every unreinforced press, inter-trial press and extinction press costs 0.5;
    the p=0.5 PREE group still nets +0.25 per press in training);
  - renewal and recovery extinguish to a criterion, as animal designs do: extinction trials
    continue until the trailing 10 A trials have press rate <= 0.2, at most EXT_MAX = 600
    (the trials used and whether the criterion was reached are recorded per life);
  - PREE keeps the fixed EXT 300 extinction trials (its metric is the trials to extinction).
Nothing else changed: arms, groups, predictions, metrics, seeds and the "shows" rule stand.

v2 effects, declared 2026-10-28 00:30 after reading the v1 run, before running them. Two v1
designs were confounded and are redone (v1 stays in the receipt, reported as confounded):
  pree2       In v1 the p=0.5 group barely learned to press (press rate 0.32 Cadence, 0.17 MLP
              over the last 50 training trials), so its extinction started from a low level and
              the comparison is void. A pilot (seeds 0-3, acquisition only) showed acquisition
              under p=0.5 is slow and variable at cost 0.5 and mostly reached at 0.25. So: COST2
              0.25 for this effect only (training non-reinforced presses, inter-trial presses and
              extinction presses alike, as in animal PREE where a non-reinforced training trial is
              the same event as an extinction trial). One job per seed and arm: the p=0.5 group
              trains until the trailing 20 A trials have press rate >= 0.8, then 100 more trials
              (at most 1500 before the criterion); the p=1.0 group, same seed, trains the same
              number of trials N. Then 600 extinction trials each. Metrics: trials to
              extinction (trailing 10 <= 0.2; 600 if never) and press rate over the first 50
              extinction trials. Pairs where the p=0.5 group never met the criterion are
              excluded and counted. Prediction (animals): p=0.5 extinguishes later.
  latent2     In v1 a press during pre-exposure cost 0.5, so pre-exposure punished pressing A:
              that is response training, not latent inhibition. So: during pre-exposure an A
              moment has no consequence (outcome 0 whatever the action; inter-trial moments as
              always), the no-stimulus group gets the same number of inter-trial moments, then A
              reinforced for 400 trials (v1: 200, many lives were still censored). Metrics as v1
              (press rate over the first 50 acquisition trials; trials to acquisition, 400 if
              never). Prediction (animals): pre acquires more slowly.
Same arms, seeds 0-23, COST 0.5 everywhere else, same "shows" rule.
Statistics: per-seed values, group means with the standard error, and the number of seeds in
which the predicted direction holds (paired by seed: same seed, same brain founder, so each
pair differs only by the group's history). An effect "shows" when the mean difference is in the
predicted direction by more than two standard errors of the paired difference.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAIN, EXT, PHASE, TEST, WAIT, EXT_MAX = 300, 300, 200, 10, 500, 600
BIAS, A, B, C, X, Y = range(6)
PRESS, REWARD, COST = 1, 1.0, -0.5


def obs(stimuli: tuple[int, ...], context: int) -> np.ndarray:
    x = np.zeros((1, 6))
    x[0, BIAS] = 1.0
    x[0, context] = 1.0
    for s in stimuli:
        x[0, s] = 1.0
    return x


class CadenceAgent:
    def __init__(self, seed: int, memory_choice: bool) -> None:
        import cadence as cd

        self.brain = cd.Brain.compose(6, 2, modules=(16,), seed=seed, arousal=True,
                                      memory_choice=memory_choice)
        self.pending: float | None = None
        self.refusals = 0

    def moment(self, x: np.ndarray) -> int:
        try:
            if self.pending is None:
                a = self.brain.live(x)
            else:
                a = self.brain.live(x, reward=[self.pending])
        except RuntimeError:
            self.refusals += 1
            a = self.brain.live(x)
        return int(a[0])

    def outcome(self, r: float) -> None:
        self.pending = r

    def wait(self, x: np.ndarray, n: int) -> None:
        for _ in range(n):
            self.brain.wait(x)


class MLPAgent:
    def __init__(self, seed: int, eps: float = 0.1, lr: float = 0.05) -> None:
        import torch

        torch.manual_seed(seed)
        torch.set_default_dtype(torch.float64)
        self.rng = np.random.default_rng(seed + 7)
        self.net = torch.nn.Sequential(torch.nn.Linear(6, 16), torch.nn.Tanh(), torch.nn.Linear(16, 2))
        self.opt = torch.optim.SGD(self.net.parameters(), lr=lr)
        self.eps = eps
        self.last: tuple[np.ndarray, int] | None = None
        self.refusals = 0

    def moment(self, x: np.ndarray) -> int:
        import torch

        with torch.no_grad():
            q = self.net(torch.as_tensor(x))[0]
        a = int(self.rng.integers(2)) if self.rng.random() < self.eps else int(torch.argmax(q))
        self.last = (x, a)
        return a

    def outcome(self, r: float) -> None:
        import torch

        x, a = self.last
        loss = (self.net(torch.as_tensor(x))[0, a] - r) ** 2
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def wait(self, x: np.ndarray, n: int) -> None:
        pass


def make(arm: str, seed: int):
    if arm == "mlp":
        return MLPAgent(seed)
    return CadenceAgent(seed, memory_choice=(arm == "cadence_mc"))


def trials(agent, n: int, stimuli: tuple[int, ...] | None, context: int, p: float,
           rng: np.random.Generator) -> list[int]:
    """n trials: a stimulus moment (if stimuli) then an inter-trial moment. A press on the
    stimulus is reinforced with probability p (p=0: extinction). Returns presses on stimuli."""
    presses = []
    blank = obs((), context)
    for _ in range(n):
        if stimuli is not None:
            a = agent.moment(obs(stimuli, context))
            paid = a == PRESS and rng.random() < p
            agent.outcome(REWARD if paid else (COST if a == PRESS else 0.0))
            presses.append(a)
        a = agent.moment(blank)
        agent.outcome(COST if a == PRESS else 0.0)
    return presses


def extinguish(agent, stimuli: tuple[int, ...], context: int, rng: np.random.Generator) -> list[int]:
    """Extinction trials until the trailing 10 presses on the stimulus are <= 0.2, at most EXT_MAX."""
    presses: list[int] = []
    while len(presses) < EXT_MAX:
        presses += trials(agent, 1, stimuli, context, 0.0, rng)
        if len(presses) >= 10 and np.mean(presses[-10:]) <= 0.2:
            break
    return presses


def preexpose(agent, n: int, context: int) -> None:
    """n pre-exposure trials of A without consequence: the A moment's outcome is 0 whatever the
    action; the inter-trial moment as always."""
    blank = obs((), context)
    for _ in range(n):
        agent.moment(obs((A,), context))
        agent.outcome(0.0)
        a = agent.moment(blank)
        agent.outcome(COST if a == PRESS else 0.0)


def first_run(presses: list[int], level: float, above: bool, window: int = 10) -> int | None:
    v = np.asarray(presses, float)
    if len(v) < window:
        return None
    trail = np.convolve(v, np.ones(window) / window, mode="valid")
    hit = np.flatnonzero(trail >= level if above else trail <= level)
    return int(hit[0] + window) if len(hit) else None


def rate(presses: list[int]) -> float:
    return float(np.mean(presses)) if presses else float("nan")


def life(effect: str, group: str, arm: str, seed: int) -> dict:
    import warnings

    warnings.simplefilter("ignore")
    agent = make(arm, seed)
    rng = np.random.default_rng(1000 + seed)
    out: dict = {"effect": effect, "group": group, "arm": arm, "seed": seed}
    if effect == "pree":
        p = {"p1": 1.0, "p05": 0.5}[group]
        train = trials(agent, TRAIN, (A,), X, p, rng)
        ext = trials(agent, EXT, (A,), X, 0.0, rng)
        out.update(train_late=rate(train[-50:]), test=rate(ext[:50]),
                   to_extinction=first_run(ext, 0.2, above=False) or EXT)
    elif effect == "blocking":
        trials(agent, PHASE, (A,) if group == "block" else (C,), X, 1.0, rng)
        p2 = trials(agent, PHASE, (A, B), X, 1.0, rng)
        probe_a = None
        if group == "block":
            probe_a = rate(trials(copy.deepcopy(agent), TEST, (A,), X, 0.0, np.random.default_rng(seed)))
        test = trials(agent, TEST, (B,), X, 0.0, rng)
        out.update(train_late=rate(p2[-50:]), test=rate(test), probe_a=probe_a)
    elif effect == "renewal":
        acq_ctx, ext_ctx, test_ctx = {"ABA": (X, Y, X), "AAA": (X, X, X), "ABB": (X, Y, Y)}[group]
        acq = trials(agent, PHASE, (A,), acq_ctx, 1.0, rng)
        ext = extinguish(agent, (A,), ext_ctx, rng)
        test = trials(agent, TEST, (A,), test_ctx, 0.0, rng)
        out.update(train_late=rate(acq[-50:]), ext_trials=len(ext), ext_reached=len(ext) < EXT_MAX
                   or float(np.mean(ext[-10:])) <= 0.2, test=rate(test))
    elif effect == "recovery":
        acq = trials(agent, PHASE, (A,), X, 1.0, rng)
        ext = extinguish(agent, (A,), X, rng)
        if group == "later":
            # the last inter-trial action awaits its outcome while time passes; the next
            # live moment delivers it, as the outcome that was always coming
            agent.wait(obs((), X), WAIT)
        test = trials(agent, TEST, (A,), X, 0.0, rng)
        out.update(train_late=rate(acq[-50:]), ext_trials=len(ext), ext_reached=len(ext) < EXT_MAX
                   or float(np.mean(ext[-10:])) <= 0.2, test=rate(test))
    elif effect == "latent":
        trials(agent, PHASE, (A,) if group == "pre" else None, X, 0.0, rng)
        acq = trials(agent, PHASE, (A,), X, 1.0, rng)
        out.update(test=rate(acq[:50]), to_acquire=first_run(acq, 0.8, above=True) or PHASE,
                   train_late=rate(acq[-50:]))
    elif effect == "latent2":
        if group == "pre":
            preexpose(agent, PHASE, X)
        else:
            trials(agent, PHASE, None, X, 0.0, rng)
        acq = trials(agent, 400, (A,), X, 1.0, rng)
        out.update(test=rate(acq[:50]), to_acquire=first_run(acq, 0.8, above=True) or 400,
                   train_late=rate(acq[-50:]))
    elif effect == "pree2":
        return pree2(arm, seed)
    else:
        raise ValueError(effect)
    out["refusals"] = agent.refusals
    return out


def pree2(arm: str, seed: int) -> dict:
    global COST
    saved, COST = COST, -0.25
    try:
        agent = make(arm, seed)
        rng = np.random.default_rng(1000 + seed)
        train: list[int] = []
        reached = None
        while len(train) < 1500 and reached is None:
            train += trials(agent, 1, (A,), X, 0.5, rng)
            if len(train) >= 20 and np.mean(train[-20:]) >= 0.8:
                reached = len(train)
        n = len(train)
        if reached is not None:
            train += trials(agent, 100, (A,), X, 0.5, rng)
            n = len(train)
        ext05 = trials(agent, 600, (A,), X, 0.0, rng)
        other = make(arm, seed)
        rng1 = np.random.default_rng(1000 + seed)
        train1 = trials(other, n, (A,), X, 1.0, rng1)
        ext1 = trials(other, 600, (A,), X, 0.0, rng1)
        return {"effect": "pree2", "group": "pair", "arm": arm, "seed": seed, "reached": reached,
                "train_trials": n, "train_late05": rate(train[-50:]), "train_late1": rate(train1[-50:]),
                "test05": rate(ext05[:50]), "test1": rate(ext1[:50]),
                "to_ext05": first_run(ext05, 0.2, above=False) or 600,
                "to_ext1": first_run(ext1, 0.2, above=False) or 600,
                "refusals": agent.refusals + other.refusals}
    finally:
        COST = saved


GROUPS = {
    "pree": ("p1", "p05"),
    "blocking": ("block", "control"),
    "renewal": ("ABA", "AAA", "ABB"),
    "recovery": ("now", "later"),
    "latent": ("pre", "none"),
    "latent2": ("pre", "none"),
    "pree2": ("pair",),
}
# (effect, group expected higher on `metric`, group expected lower, metric)
PREDICTIONS = [
    ("pree", "p05", "p1", "test"),
    ("pree", "p05", "p1", "to_extinction"),
    ("blocking", "control", "block", "test"),
    ("renewal", "ABA", "AAA", "test"),
    ("renewal", "ABA", "ABB", "test"),
    ("recovery", "later", "now", "test"),
    ("latent", "none", "pre", "test"),
    ("latent", "pre", "none", "to_acquire"),
    ("latent2", "none", "pre", "test"),
    ("latent2", "pre", "none", "to_acquire"),
]
ARMS = ("cadence", "cadence_mc", "mlp")


def _job(spec: tuple) -> dict:
    try:
        return life(*spec)
    except Exception as error:  # noqa: BLE001
        e, g, a, s = spec
        return {"effect": e, "group": g, "arm": a, "seed": s, "error": f"{type(error).__name__}: {error}"[:300]}


def summarize(rows: list[dict]) -> dict:
    by = {}
    for r in rows:
        if "error" not in r:
            by[(r["effect"], r["group"], r["arm"], r["seed"])] = r
    table = []
    for effect, hi, lo, metric in PREDICTIONS:
        for arm in ARMS:
            seeds = sorted({s for (e, g, a, s) in by if e == effect and a == arm})
            pairs = [(by[(effect, hi, arm, s)][metric], by[(effect, lo, arm, s)][metric])
                     for s in seeds if (effect, hi, arm, s) in by and (effect, lo, arm, s) in by]
            if not pairs:
                continue
            h = np.array([p[0] for p in pairs], float)
            l = np.array([p[1] for p in pairs], float)
            d = h - l
            se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
            table.append({
                "effect": effect, "metric": metric, "arm": arm, "higher": hi, "lower": lo,
                "n": len(d), "mean_higher": float(h.mean()), "mean_lower": float(l.mean()),
                "diff": float(d.mean()), "se": se, "seeds_in_direction": int((d > 0).sum()),
                "ties": int((d == 0).sum()), "shows": bool(d.mean() > 2 * se) if se == se else False,
            })
    pairs2 = {}
    for r in rows:
        if r.get("effect") == "pree2" and "error" not in r:
            pairs2.setdefault(r["arm"], []).append(r)
    for arm, rs in pairs2.items():
        ok = [r for r in rs if r["reached"] is not None]
        for metric, hi, lo in (("to_ext", "to_ext05", "to_ext1"), ("test", "test05", "test1")):
            if len(ok) < 2:
                continue
            h = np.array([r[hi] for r in ok], float)
            l = np.array([r[lo] for r in ok], float)
            d = h - l
            se = float(d.std(ddof=1) / np.sqrt(len(d)))
            table.append({"effect": "pree2", "metric": metric, "arm": arm, "higher": "p05",
                          "lower": "p1", "n": len(d), "excluded": len(rs) - len(ok),
                          "mean_higher": float(h.mean()), "mean_lower": float(l.mean()),
                          "diff": float(d.mean()), "se": se, "seeds_in_direction": int((d > 0).sum()),
                          "ties": int((d == 0).sum()), "shows": bool(d.mean() > 2 * se)})
    return {"predictions": table}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(24)))
    ap.add_argument("--effects", nargs="+", default=list(GROUPS))
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT / "runs" / "animal_battery" / "receipt.json"))
    ap.add_argument("--rows", default=None, help="also write the raw rows here")
    a = ap.parse_args()
    jobs = [(e, g, arm, s) for e in a.effects for g in GROUPS[e] for arm in a.arms for s in a.seeds]
    t0 = time.time()
    with get_context("spawn").Pool(a.workers) as pool:
        rows = pool.map(_job, jobs, chunksize=1)
    seconds = round(time.time() - t0, 1)
    errors = [r for r in rows if "error" in r]
    print(f"{len(rows)} lives, {seconds}s, errors: {len(errors)}")
    for e in errors[:5]:
        print("ERROR", e)
    summary = summarize(rows)
    print("| effect | metric | arm | higher (pred.) | lower (pred.) | diff +/- se | seeds in direction | shows |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for t in summary["predictions"]:
        print(f"| {t['effect']} | {t['metric']} | {t['arm']} | {t['higher']} {t['mean_higher']:.2f} |"
              f" {t['lower']} {t['mean_lower']:.2f} | {t['diff']:+.2f} +/- {t['se']:.2f} |"
              f" {t['seeds_in_direction']}/{t['n']} ({t['ties']} ties) | {'yes' if t['shows'] else 'no'} |")
    import cadence as cd

    body = {"schema": "cadence-lab/animal-battery/v1", "cadence": cd.__version__,
            "params": {"TRAIN": TRAIN, "EXT": EXT, "PHASE": PHASE, "TEST": TEST, "WAIT": WAIT,
                       "EXT_MAX": EXT_MAX, "COST": COST},
            "seeds": a.seeds, "arms": a.arms, "effects": a.effects, "seconds": seconds,
            "rows": rows, **summary}
    if a.rows:
        Path(a.rows).parent.mkdir(parents=True, exist_ok=True)
        Path(a.rows).write_text(json.dumps(body, indent=1), encoding="utf-8", newline="\n")
    sources = [("experiments/animal_battery.py", Path(__file__))]
    receipt = cd.Receipt.build(body["schema"], body, sources=sources)
    path = receipt.write(Path(a.out))
    ok, msg = cd.Receipt.verify(path, sources=sources)
    print(f"\nreceipt: {path} ({ok} {msg})")


if __name__ == "__main__":
    main()
