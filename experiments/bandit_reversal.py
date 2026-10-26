"""Two-armed bandit with a mid-life reversal, one continuing Cadence brain vs controls.

Run:  python experiments/bandit_reversal.py --seeds 0 1 2 3 4 5 6 7
Writes runs/bandit_reversal/receipt.json and prints a summary table.

Arms (same stream, same seeds, one observation, one outcome per moment):
  continuing  Brain.compose(1, 2, modules=(16,), arousal=True) lives through the reversal.
  cold_reset  same brain, but brain.reset(keep=()) at the reversal (arousal state cleared).
  newborn     a fresh brain (new seed) replaces the old one at the reversal.
  memory_off  same as continuing but episodic=False (no associative store).
  mlp         small PyTorch MLP (1-16-2) Q-learner, epsilon-greedy, one SGD step per moment.
  random      uniform random arm (chance control).

Metrics per arm: hit rate (paying arm chosen) in the last window before the reversal,
the first window after it and the last window of life; moments to recover (first moment
after the reversal where the trailing 20-moment hit rate reaches 0.9); work counters.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cadence as cd
from cadence import Brain

from envs.bandit import ReversalBandit

WINDOW = 50
RECOVER_TRAIL = 20
RECOVER_LEVEL = 0.9


def brain_work(brain: Brain) -> dict:
    a = brain.arousal
    return {
        "routine_moments": int(a.moments.get("routine", 0)),
        "aroused_moments": int(a.moments.get("aroused", 0)),
        "answer_sweeps": int(sum(a.sweeps.values())),
        "learning_sweeps": int(a.learning_sweeps),
        "learner_updates": int(brain.learner.updates),
    }


def add_work(total: dict, part: dict) -> dict:
    return {k: total.get(k, 0) + v for k, v in part.items()}


def make_brain(seed: int, *, episodic: bool = True) -> Brain:
    return Brain.compose(1, 2, modules=(16,), seed=seed, arousal=True, episodic=episodic)


def live(brain: Brain, obs, reward=None, refusals=None):
    """One live moment with the documented retry: on refusal after an accepted
    outcome, retry without the outcome."""
    try:
        return brain.live(obs) if reward is None else brain.live(obs, reward=[reward])
    except RuntimeError:
        refusals[0] += 1
        return brain.live(obs)


def run_brain(arm: str, seed: int, life: int, reversal: int, p: float, back=None) -> dict:
    env = ReversalBandit(life, reversal, p, seed=seed, back=back)
    brain = make_brain(seed, episodic=(arm != "memory_off"))
    obs = env.observation
    refusals = [0]
    work: dict = {}
    hits, rewards, modes = [], [], []
    action = live(brain, obs, refusals=refusals)
    for t in range(life):
        if t == reversal and arm == "cold_reset":
            brain.reset(keep=())
            action = live(brain, obs, refusals=refusals)
        elif t == reversal and arm == "newborn":
            work = add_work(work, brain_work(brain))
            brain = make_brain(seed + 1000)
            action = live(brain, obs, refusals=refusals)
        reward, hit = env.pull(t, int(action[0]))
        hits.append(hit)
        rewards.append(reward)
        action = live(brain, obs, reward, refusals)
        modes.append(brain.last_arousal["mode"] if brain.last_arousal else "none")
    work = add_work(work, brain_work(brain))
    work["refusals"] = refusals[0]
    return {"hits": hits, "rewards": rewards, "modes": modes, "work": work}


def run_mlp(seed: int, life: int, reversal: int, p: float, back=None, eps: float = 0.1, lr: float = 0.05) -> dict:
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 7)
    env = ReversalBandit(life, reversal, p, seed=seed, back=back)
    net = torch.nn.Sequential(torch.nn.Linear(1, 16), torch.nn.Tanh(), torch.nn.Linear(16, 2))
    opt = torch.optim.SGD(net.parameters(), lr=lr)
    x = torch.ones(1, 1)
    hits, rewards = [], []
    for t in range(life):
        with torch.no_grad():
            q = net(x)[0]
        arm = int(rng.integers(2)) if rng.random() < eps else int(torch.argmax(q))
        reward, hit = env.pull(t, arm)
        hits.append(hit)
        rewards.append(reward)
        loss = (net(x)[0, arm] - reward) ** 2
        opt.zero_grad()
        loss.backward()
        opt.step()
    params = sum(p.numel() for p in net.parameters())
    return {"hits": hits, "rewards": rewards, "modes": [],
            "work": {"sgd_updates": life, "forward_passes": 2 * life, "parameters": params}}


def run_random(seed: int, life: int, reversal: int, p: float, back=None) -> dict:
    rng = np.random.default_rng(seed + 99)
    env = ReversalBandit(life, reversal, p, seed=seed, back=back)
    hits, rewards = [], []
    for t in range(life):
        reward, hit = env.pull(t, int(rng.integers(2)))
        hits.append(hit)
        rewards.append(reward)
    return {"hits": hits, "rewards": rewards, "modes": [], "work": {}}


def recover_moments(hits: list[bool], reversal: int) -> int | None:
    h = np.asarray(hits, dtype=float)
    for t in range(reversal + RECOVER_TRAIL, len(h) + 1):
        if h[t - RECOVER_TRAIL:t].mean() >= RECOVER_LEVEL:
            return t - reversal
    return None


def summarize(run: dict, reversal: int, back=None) -> dict:
    h = np.asarray(run["hits"], dtype=float)
    r = np.asarray(run["rewards"], dtype=float)
    out = {
        "pre_hit": float(h[reversal - WINDOW:reversal].mean()),
        "post_first_hit": float(h[reversal:reversal + WINDOW].mean()),
        "last_hit": float(h[-WINDOW:].mean()),
        "post_reward_mean": float(r[reversal:].mean()),
        "recover_moments": recover_moments(run["hits"], reversal),
        "window_hits": [float(h[i:i + WINDOW].mean()) for i in range(0, len(h), WINDOW)],
    }
    if back is not None:
        out["pre_hit"] = float(h[reversal - WINDOW:reversal].mean())
        out["post_first_hit"] = float(h[reversal:reversal + WINDOW].mean())
        out["post_reward_mean"] = float(r[reversal:back].mean())
        out["recover_moments"] = recover_moments(run["hits"][:back], reversal)
        out["back_pre_hit"] = float(h[back - WINDOW:back].mean())
        out["back_first_hit"] = float(h[back:back + WINDOW].mean())
        out["back_reward_mean"] = float(r[back:].mean())
        out["back_recover_moments"] = recover_moments(run["hits"], back)
    if run["modes"]:
        m = run["modes"][reversal:reversal + WINDOW]
        out["post_first_aroused_share"] = m.count("aroused") / len(m)
    return out


ARMS = ("continuing", "cold_reset", "newborn", "memory_off", "mlp", "random")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    ap.add_argument("--life", type=int, default=600)
    ap.add_argument("--reversal", type=int, default=300)
    ap.add_argument("--p", type=float, default=1.0, help="reward reliability")
    ap.add_argument("--back", type=int, default=None, help="second flip back to arm 0 (A-B-A)")
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--out", default=str(ROOT / "runs" / "bandit_reversal" / "receipt.json"))
    args = ap.parse_args()

    rows = []
    for arm in args.arms:
        for seed in args.seeds:
            t0 = time.perf_counter()
            if arm == "mlp":
                run = run_mlp(seed, args.life, args.reversal, args.p, args.back)
            elif arm == "random":
                run = run_random(seed, args.life, args.reversal, args.p, args.back)
            else:
                run = run_brain(arm, seed, args.life, args.reversal, args.p, args.back)
            elapsed = time.perf_counter() - t0
            row = {"arm": arm, "seed": seed, "elapsed_s": round(elapsed, 3),
                   **summarize(run, args.reversal, args.back), "work": run["work"]}
            rows.append(row)
            print(f"{arm:11s} seed {seed}: pre {row['pre_hit']:.2f} post50 {row['post_first_hit']:.2f} "
                  f"last {row['last_hit']:.2f} recover {row['recover_moments']} ({elapsed:.1f}s)", flush=True)

    table = {}
    for arm in args.arms:
        rs = [r for r in rows if r["arm"] == arm]
        rec = [r["recover_moments"] for r in rs]
        done = [x for x in rec if x is not None]
        table[arm] = {
            "seeds": len(rs),
            "pre_hit_mean": float(np.mean([r["pre_hit"] for r in rs])),
            "post_first_hit_mean": float(np.mean([r["post_first_hit"] for r in rs])),
            "last_hit_mean": float(np.mean([r["last_hit"] for r in rs])),
            "recovered": f"{len(done)}/{len(rs)}",
            "recover_median": float(np.median(done)) if done else None,
            "elapsed_s_mean": float(np.mean([r["elapsed_s"] for r in rs])),
        }
        if args.back is not None:
            brec = [r["back_recover_moments"] for r in rs if r["back_recover_moments"] is not None]
            table[arm] |= {"back_first_hit_mean": float(np.mean([r["back_first_hit"] for r in rs])),
                           "back_recovered": f"{len(brec)}/{len(rs)}",
                           "back_recover_median": float(np.median(brec)) if brec else None}
        if rs and "answer_sweeps" in rs[0]["work"]:
            table[arm]["sweeps_mean"] = float(np.mean(
                [r["work"]["answer_sweeps"] + r["work"]["learning_sweeps"] for r in rs]))
            table[arm]["aroused_share"] = float(np.mean(
                [r["work"]["aroused_moments"] / max(1, r["work"]["aroused_moments"] + r["work"]["routine_moments"])
                 for r in rs]))

    print("\narm          pre   post50  last  recovered  median  sweeps")
    for arm, s in table.items():
        print(f"{arm:11s} {s['pre_hit_mean']:.2f}  {s['post_first_hit_mean']:.2f}   {s['last_hit_mean']:.2f}  "
              f"{s['recovered']:>9s}  {s['recover_median']}  {s.get('sweeps_mean', '-')}"
              + (f"  | back: first50 {s['back_first_hit_mean']:.2f} recovered {s['back_recovered']} "
                 f"median {s['back_recover_median']}" if args.back is not None else ""))

    import torch
    body = {
        "protocol": {"life": args.life, "reversal": args.reversal, "p": args.p, "seeds": args.seeds,
                     "arms": args.arms, "window": WINDOW, "recover_trail": RECOVER_TRAIL,
                     "recover_level": RECOVER_LEVEL, "back": args.back,
                     "brain": "Brain.compose(1, 2, modules=(16,), arousal=True), backend cpu float64",
                     "reward": "+1 paying arm / -1 other (flipped with prob 1-p)"},
        "versions": {"cadence": "0.89.0", "numpy": np.__version__, "torch": torch.__version__,
                     "python": sys.version.split()[0]},
        "rows": rows,
        "table": table,
        "boundary": "One small deterministic-or-noisy bandit. No claim beyond this task and these seeds.",
    }
    sources = [("experiments/bandit_reversal.py", Path(__file__)),
               ("envs/bandit.py", ROOT / "envs" / "bandit.py")]
    receipt = cd.Receipt.build("cadence-lab/bandit-reversal/v1", body, sources=sources)
    path = receipt.write(Path(args.out))
    _ok, msg = cd.Receipt.verify(path, sources=sources)
    print(f"\nreceipt: {path} ({msg})")


if __name__ == "__main__":
    main()
