"""Serial reversal: does a Cadence brain get faster at reversals with practice (learning to learn)?

Run:  python experiments/serial_reversal.py --seeds 0 1 2 3 4 5 6 7 8 9 10 11
Writes runs/serial_reversal/receipt.json and prints a per-reversal table.

Animals in serial reversal get faster each flip (Harlow's learning sets; macaques end near one
error per reversal). Plain deep nets trained continually tend to stay flat or slow down
(loss of plasticity). This asks which way a Cadence brain goes.

The world is envs/serial.py: two arms, the paying arm flips `reversals` times, blocks jittered so
the flips can't be timed. Arms (same schedule and reward stream per seed):
  continuing     Brain.compose(1, 2, modules=(16,), arousal=True), constant observation.
  memory_choice  same, with memory_choice=True (episodic place + context memory votes in the choice).
  memory_off     same as continuing, episodic=False.
  newborn        a fresh brain replaces the old one at every flip.
  cue            continuing, but the observation also carries the last arm and last outcome
                 ([1, arm0, arm1, won]) so a win-stay/lose-shift rule is learnable at all.
  cue_choice     cue + memory_choice.
  mlp            1-16-2 Q-learner as in bandit_reversal.py.
  mlp_cue        the MLP on the cue observation.
  mlp_cbp        the MLP with continual-backprop-style resets (Dohare et al. 2024): every 25
                 moments the least useful hidden unit is reinitialized.
  wsls           hand-coded win-stay/lose-shift (the ideal learning-set strategy, floor reference).
  random         uniform random arm.

Per reversal: errors in the first 20 moments after the flip (perseverative errors) and moments
to recover (trailing 20-moment hit rate reaches 0.9). Per arm: mean errors per reversal index and
the slope of errors over reversals 1..N (negative = getting faster = learning to learn).
"""

from __future__ import annotations

import argparse
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
from bandit_reversal import brain_work, add_work, live

ERR_WINDOW = 20
RECOVER_TRAIL = 20
RECOVER_LEVEL = 0.9


def cue_obs(last_arm: int | None, last_reward: float | None) -> np.ndarray:
    if last_arm is None:
        return np.array([[1.0, 0.0, 0.0, 0.5]])
    return np.array([[1.0, float(last_arm == 0), float(last_arm == 1), 1.0 if last_reward > 0 else 0.0]])


def make_brain(arm: str, seed: int) -> Brain:
    inputs = 4 if arm.startswith("cue") else 1
    kw = {}
    if arm in ("memory_choice", "cue_choice"):
        kw["memory_choice"] = True
    return Brain.compose(inputs, 2, modules=(16,), seed=seed, arousal=True,
                         episodic=(arm != "memory_off"), **kw)


def run_brain(arm: str, env: SerialBandit, seed: int) -> dict:
    brain = make_brain(arm, seed)
    cued = arm.startswith("cue")
    obs = cue_obs(None, None) if cued else env.observation
    refusals = [0]
    work: dict = {}
    hits, modes = [], []
    born = 0
    action = live(brain, obs, refusals=refusals)
    for t in range(env.life):
        if arm == "newborn" and t in env.flips:
            work = add_work(work, brain_work(brain))
            born += 1
            brain = make_brain(arm, seed + 1000 * born)
            action = live(brain, obs, refusals=refusals)
        a = int(action[0])
        reward, hit = env.pull(t, a)
        hits.append(hit)
        if cued:
            # the outcome is given for the moment just acted on, the cue for the next one
            brain_reward = reward
            obs = cue_obs(a, reward)
            action = live(brain, obs, brain_reward, refusals)
        else:
            action = live(brain, obs, reward, refusals)
        modes.append(brain.last_arousal["mode"] if brain.last_arousal else "none")
    work = add_work(work, brain_work(brain))
    work["refusals"] = refusals[0]
    return {"hits": hits, "modes": modes, "work": work}


def run_mlp(arm: str, env: SerialBandit, seed: int, eps: float = 0.1, lr: float = 0.05,
            cbp_every: int = 25) -> dict:
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 7)
    cued = arm == "mlp_cue"
    n_in = 4 if cued else 1
    l1, l2 = torch.nn.Linear(n_in, 16), torch.nn.Linear(16, 2)
    net = torch.nn.Sequential(l1, torch.nn.Tanh(), l2)
    opt = torch.optim.SGD(net.parameters(), lr=lr)
    x = torch.tensor(cue_obs(None, None) if cued else env.observation, dtype=torch.float32)
    utility = torch.zeros(16)
    resets = 0
    hits = []
    for t in range(env.life):
        with torch.no_grad():
            q = net(x)[0]
        a = int(rng.integers(2)) if rng.random() < eps else int(torch.argmax(q))
        reward, hit = env.pull(t, a)
        hits.append(hit)
        h = torch.tanh(l1(x))[0]
        loss = (l2(h.unsqueeze(0))[0, a] - reward) ** 2
        opt.zero_grad()
        loss.backward()
        opt.step()
        if arm == "mlp_cbp":
            with torch.no_grad():
                # contribution utility: |activation| * summed |outgoing weight|, running average
                utility = 0.99 * utility + 0.01 * h.detach().abs() * l2.weight.abs().sum(0)
                if (t + 1) % cbp_every == 0:
                    u = int(torch.argmin(utility))
                    fresh = torch.nn.Linear(n_in, 16)
                    l1.weight[u] = fresh.weight[u]
                    l1.bias[u] = 0.0
                    l2.weight[:, u] = 0.0
                    utility[u] = float(utility.median())
                    resets += 1
        if cued:
            x = torch.tensor(cue_obs(a, reward), dtype=torch.float32)
    return {"hits": hits, "modes": [], "work": {"sgd_updates": env.life, "unit_resets": resets}}


def run_simple(arm: str, env: SerialBandit, seed: int) -> dict:
    rng = np.random.default_rng(seed + 99)
    hits = []
    a, last_reward = 0, 1.0
    for t in range(env.life):
        if arm == "random":
            a = int(rng.integers(2))
        elif last_reward <= 0:  # wsls
            a = 1 - a
        last_reward, hit = env.pull(t, a)
        hits.append(hit)
    return {"hits": hits, "modes": [], "work": {}}


def per_reversal(hits: list[bool], env: SerialBandit) -> list[dict]:
    h = np.asarray(hits, dtype=float)
    ends = env.flips[1:] + [env.life]
    out = []
    for k, (f, end) in enumerate(zip(env.flips, ends), start=1):
        rec = None
        for t in range(f + RECOVER_TRAIL, end + 1):
            if h[t - RECOVER_TRAIL:t].mean() >= RECOVER_LEVEL:
                rec = t - f
                break
        out.append({"reversal": k, "errors": int(ERR_WINDOW - h[f:f + ERR_WINDOW].sum()),
                    "recover": rec, "block_hit": float(h[f:end].mean())})
    return out


def slope(ys: list[float]) -> float:
    x = np.arange(1, len(ys) + 1, dtype=float)
    return float(np.polyfit(x, np.asarray(ys, dtype=float), 1)[0])


ARMS = ("continuing", "memory_choice", "memory_off", "newborn", "cue", "cue_choice",
        "mlp", "mlp_cue", "mlp_cbp", "wsls", "random")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(12)))
    ap.add_argument("--reversals", type=int, default=10)
    ap.add_argument("--first", type=int, default=200)
    ap.add_argument("--block", type=int, default=100)
    ap.add_argument("--jitter", type=int, default=20)
    ap.add_argument("--p", type=float, default=1.0)
    ap.add_argument("--fail", type=float, default=-1.0, help="reward for the unpaid arm (0 = issue #210 case)")
    ap.add_argument("--probe", type=int, default=None, help="extra long block before one extra flip")
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--out", default=str(ROOT / "runs" / "serial_reversal" / "receipt.json"))
    args = ap.parse_args()

    rows = []
    for arm in args.arms:
        for seed in args.seeds:
            env = SerialBandit(args.reversals, args.first, args.block, args.jitter, args.p, seed=seed,
                              fail=args.fail, probe=args.probe)
            t0 = time.perf_counter()
            if arm.startswith("mlp"):
                run = run_mlp(arm, env, seed)
            elif arm in ("wsls", "random"):
                run = run_simple(arm, env, seed)
            else:
                run = run_brain(arm, env, seed)
            elapsed = time.perf_counter() - t0
            revs = per_reversal(run["hits"], env)
            h = np.asarray(run["hits"], dtype=float)
            row = {"arm": arm, "seed": seed, "elapsed_s": round(elapsed, 3), "flips": env.flips,
                   "life": env.life, "first_block_hit": float(h[:env.flips[0]].mean()),
                   "reversals": revs, "error_slope": slope([r["errors"] for r in revs]),
                   "work": run["work"]}
            rows.append(row)
            errs = " ".join(f"{r['errors']:2d}" for r in revs)
            print(f"{arm:13s} seed {seed:2d}: errors [{errs}] slope {row['error_slope']:+.2f} ({elapsed:.1f}s)",
                  flush=True)

    table = {}
    for arm in args.arms:
        rs = [r for r in rows if r["arm"] == arm]
        errs = np.array([[v["errors"] for v in r["reversals"]] for r in rs], dtype=float)
        recs = [[v["recover"] for v in r["reversals"]] for r in rs]
        slopes = [r["error_slope"] for r in rs]
        table[arm] = {
            "seeds": len(rs),
            "errors_by_reversal": [float(x) for x in errs.mean(0)],
            "errors_first3": float(errs[:, :3].mean()),
            "errors_last3": float(errs[:, -3:].mean()),
            "slope_mean": float(np.mean(slopes)),
            "slope_negative": int(sum(s < 0 for s in slopes)),
            "recover_median_by_reversal": [
                (float(np.median(d)) if (d := [x[k] for x in recs if x[k] is not None]) else None)
                for k in range(len(recs[0]))],
            "recovered_share": float(np.mean([x is not None for rec in recs for x in rec])),
            "first_block_hit": float(np.mean([r["first_block_hit"] for r in rs])),
            "elapsed_s_mean": float(np.mean([r["elapsed_s"] for r in rs])),
        }

    print(f"\n{'arm':13s} first3  last3  slope/rev  neg-slope  recovered  errors by reversal")
    for arm, s in table.items():
        eb = " ".join(f"{x:4.1f}" for x in s["errors_by_reversal"])
        print(f"{arm:13s} {s['errors_first3']:6.2f} {s['errors_last3']:6.2f}  {s['slope_mean']:+8.3f}  "
              f"{s['slope_negative']:>4d}/{s['seeds']:<4d}  {s['recovered_share']:8.2f}   {eb}")

    import torch
    body = {
        "protocol": {"reversals": args.reversals, "first": args.first, "block": args.block,
                     "jitter": args.jitter, "p": args.p,
                     "fail": args.fail, "probe": args.probe, "seeds": args.seeds, "arms": args.arms,
                     "err_window": ERR_WINDOW, "recover_trail": RECOVER_TRAIL,
                     "recover_level": RECOVER_LEVEL,
                     "brain": "Brain.compose(n, 2, modules=(16,), arousal=True), backend cpu float64",
                     "reward": "+1 paying arm / `fail` other (flipped with prob 1-p)"},
        "versions": {"cadence": cd.__version__ if hasattr(cd, "__version__") else "unknown",
                     "numpy": np.__version__, "torch": torch.__version__,
                     "python": sys.version.split()[0]},
        "rows": rows,
        "table": table,
        "boundary": "One small two-armed serial-reversal task. No claim beyond this task and these seeds.",
    }
    sources = [("experiments/serial_reversal.py", Path(__file__)),
               ("envs/serial.py", ROOT / "envs" / "serial.py")]
    receipt = cd.Receipt.build("cadence-lab/serial-reversal/v1", body, sources=sources)
    path = receipt.write(Path(args.out))
    _ok, msg = cd.Receipt.verify(path, sources=sources)
    print(f"\nreceipt: {path} ({msg})")


if __name__ == "__main__":
    main()
