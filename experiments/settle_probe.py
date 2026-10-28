"""Settle probe: does a Cadence brain's settled state carry the ball's direction in One-Frame Breakout?

Run with the classic venv (cadence-net 0.89.0, minatar, scipy):

    P:/projects/cadence-arcade/.venv-classic/Scripts/python.exe experiments/settle_probe.py \
        --arms norecall,pos32 --seeds 10-13 --procs 2
    ... experiments/settle_probe.py --summarize

PRE-DECLARED (written and committed before any probe was run)

Background. In cadence-arcade/classic (One-Frame Breakout) the player sees one frame of hand
features (ball column, ball row, paddle column, ball-minus-paddle), so which way the ball moves is
only knowable from a moment ago. After 300 blocks every Cadence arm, recall off included, reaches
~13 bricks per test game while a one-frame MLP stops near 6 (notes/long_training_breakout.md).
Hypothesis: the brain's settled state carries the previous moment, i.e. the ball's direction.

What the brain has (read from cadence 0.89.0 source, not assumed). `Brain.compose(49, 3,
modules=(32,), episodic=False)` builds sensory 0-48, association 49-80, prefrontal 81-112, motor
113-115, and a `working_memory` Trace (decay 0, amplitude 3) that copies the association
activation of the last accepted moment into the prefrontal region as next moment's drive. Each
settle also starts warm from the last settled state (`basal_ganglia.state`). So "the settle
carries the previous moment" has two candidate carriers: the explicit prefrontal copy and the warm
start. "norecall" means no recall slots; the working-memory copy is still on.

Training. Players are built and trained exactly as classic/run.py `life` does (imported, not
re-implemented: players.CadenceStudent, breakout.Breakout, run.test; same seeds, 1,000 training
moments per block, a 3-episode test per block). The block test scores must reproduce the long
run (runs/long10x) for the same seed, which checks that the probe did not disturb training.
Arms: norecall (main), pos32 (recall on, position term on; secondary). Seeds 10-13. 150 blocks.
Checkpoints: before training (block 0, the untrained brain), and after 10, 30, 75, 150 blocks.

Probe data. At each checkpoint a deep copy of the player is frozen (play only, no lessons) and
plays fresh probe games (env seed from (seed, block, 99)) until >= 3,000 moments and >= 10
episodes (max 150 episodes, 300 moments each). At every moment, before acting: the ball's
position now and a moment ago (MinAtar `last_x/last_y`), MinAtar's `ball_dir` (the next move),
the 49 input features, the features of the previous frame (MinAtar's trail), the working-memory
trace; after acting: the settled activations of all 116 neurons.

Targets. Main: dir4 = (sign(ball_x - last_x), sign(ball_y - last_y)), the direction the ball
just moved, over moments where both differ from 0 (episode starts and wall-clamped moments are
dropped and counted). Secondary: dx and dy alone, and MinAtar's ball_dir (next move, all moments).

Probe. Multinomial logistic regression, features standardized on the training folds, L2 1e-3,
L-BFGS. 5-fold cross-validation grouped by episode (no episode in both train and test).
Reported: held-out accuracy, and the majority-class baseline from the training folds.
Feature sets:
  frame        the 49 input features (control: one frame, direction should be near chance)
  two_frame    frame + previous frame (upper bound: direction is linear in these)
  state        settled association + prefrontal + motor activations (67)
  assoc, pfc   the association / prefrontal block alone
  frame_state  frame + state
  frame_trace  frame + the working-memory trace read before the act
Untrained control: the same probes at checkpoint 0 (the same seed's brain before any lesson).

Lesions (causal check, test time only, the trained weights untouched). At each checkpoint the
frozen copy plays 10 test episodes (env seed from (seed, block, 55)) under four conditions:
  intact; wm0 (working-memory trace zeroed before every act); cold (settle starts from rest every
  moment: the warm start removed, trace kept); both. Metric: bricks per episode.

Predictions (norecall, 150 blocks unless stated):
  P1  state probe on dir4 beats the frame probe by >= 15 points, in 4/4 seeds.
  P2  frame_trace beats frame by >= 15 points in 4/4 seeds (the prefrontal copy carries it).
  P3  untrained brain: frame_state also beats frame (the copy is wired in by construction, a
      random reservoir of the last frame), but by less than at 150 blocks in >= 3/4 seeds.
  P4  state-probe dir4 accuracy at 150 >= at 30 in >= 3/4 seeds, and across seeds x trained
      checkpoints (10, 30, 75, 150) the Spearman correlation between state-probe accuracy and
      intact lesion-test bricks is > 0.
  P5  wm0 drops bricks by > 30% vs intact in 4/4 seeds, toward the one-frame MLP level (~6).
      cold costs less than wm0.
  Null: if state <= frame + 5 points on dir4, the settled state does not linearly carry the
  direction and the 13-vs-6 gap needs another explanation. That would be the result.
pos32 is reported alongside with no separate predictions (its recall slots are a third carrier).
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLASSIC = Path("P:/projects/cadence-arcade/classic")
OUT = ROOT / "runs" / "settle_probe"
CHECKPOINTS = (0, 10, 30, 75, 150)
PROBE_MOMENTS, PROBE_MIN_EPS, PROBE_MAX_EPS = 3000, 10, 150
LESION_EPS = 10
L2 = 1e-3
FOLDS = 5
ASSOC, PFC, MOTOR = slice(49, 81), slice(81, 113), slice(113, 116)


def _write(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def _classic():
    if str(CLASSIC) not in sys.path:
        sys.path.insert(0, str(CLASSIC))
    import breakout
    import run
    return breakout, run


def collect(player, seed: int, block: int) -> dict:
    """Frozen copy plays probe games; record the env truth and the brain's state per moment."""
    breakout, run = _classic()
    p = copy.deepcopy(player)
    env = breakout.Breakout(seed=int(np.random.SeedSequence([seed, block, 99]).generate_state(1)[0]))
    rec = {k: [] for k in ("bx", "by", "lx", "ly", "dir", "ep", "frame", "prev", "trace", "act",
                           "agree")}
    n = eps = 0
    while (n < PROBE_MOMENTS or eps < PROBE_MIN_EPS) and eps < PROBE_MAX_EPS:
        env.reset(); p.new_episode(); k = 0
        while True:
            g = env.g
            obs, prev = env.observe(), env.previous()
            wm = p.brain.working_memory
            trace = wm.trace[0].copy() if wm is not None and len(wm.trace) else np.zeros(32)
            label = env.coach()
            a = p.play(obs, prev)
            st = p.brain.basal_ganglia.state
            rec["bx"].append(int(g.ball_x)); rec["by"].append(int(g.ball_y))
            rec["lx"].append(int(g.last_x)); rec["ly"].append(int(g.last_y))
            rec["dir"].append(int(g.ball_dir)); rec["ep"].append(eps)
            rec["frame"].append(obs); rec["prev"].append(prev); rec["trace"].append(trace)
            rec["act"].append(np.asarray(st.activation).reshape(-1).copy())
            rec["agree"].append(int(a == label))
            _, done = env.step(a); k += 1; n += 1
            if done or k >= run.CAP:
                break
        eps += 1
    return {k: np.asarray(v) for k, v in rec.items()}


def lesion_scores(player, seed: int, block: int) -> dict:
    breakout, run = _classic()
    out = {}
    for cond in ("intact", "wm0", "cold", "both"):
        p = copy.deepcopy(player)
        env = breakout.Breakout(seed=int(np.random.SeedSequence([seed, block, 55]).generate_state(1)[0]))
        scores = []
        for _ in range(LESION_EPS):
            env.reset(); p.new_episode(); s = k = 0
            while True:
                if cond in ("wm0", "both") and p.brain.working_memory is not None:
                    p.brain.working_memory.trace[:] = 0.0
                if cond in ("cold", "both"):
                    p.brain.basal_ganglia._free = None  # settle from rest, not from last state
                a = p.play(env.observe(), env.previous())
                r, done = env.step(a); s += r; k += 1
                if done or k >= run.CAP:
                    break
            scores.append(s)
        out[cond] = scores
    return out


# -- probe ------------------------------------------------------------------------------------

def _fit_predict(Xtr, ytr, Xte, K):
    from scipy.optimize import minimize
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-6] = 1.0
    A = np.hstack([(Xtr - mu) / sd, np.ones((len(Xtr), 1))])
    B = np.hstack([(Xte - mu) / sd, np.ones((len(Xte), 1))])
    d = A.shape[1]
    Y = np.eye(K)[ytr]

    def f(w):
        W = w.reshape(d, K)
        Z = A @ W
        Z -= Z.max(1, keepdims=True)
        P = np.exp(Z); P /= P.sum(1, keepdims=True)
        loss = -np.mean(np.log(np.maximum((P * Y).sum(1), 1e-300))) + L2 * np.sum(W[:-1] ** 2)
        G = A.T @ (P - Y) / len(A)
        G[:-1] += 2 * L2 * W[:-1]
        return loss, G.ravel()

    w = minimize(f, np.zeros(d * K), jac=True, method="L-BFGS-B", options={"maxiter": 500}).x
    return np.argmax(B @ w.reshape(d, K), 1)


def probe(X, y, groups) -> dict:
    classes, y = np.unique(y, return_inverse=True)
    K = len(classes)
    folds = groups % FOLDS
    hits = base = tot = 0
    for f in range(FOLDS):
        te, tr = folds == f, folds != f
        if te.sum() == 0 or tr.sum() == 0:
            continue
        if K == 1:
            pred = np.zeros(te.sum(), int)
        else:
            pred = _fit_predict(X[tr], y[tr], X[te], K)
        maj = np.bincount(y[tr], minlength=K).argmax()
        hits += int((pred == y[te]).sum()); base += int((y[te] == maj).sum()); tot += int(te.sum())
    return {"acc": round(hits / tot, 4), "majority": round(base / tot, 4), "n": tot, "classes": int(K)}


def analyse(rec: dict) -> dict:
    dx = np.sign(rec["bx"] - rec["lx"]); dy = np.sign(rec["by"] - rec["ly"])
    keep = (dx != 0) & (dy != 0)
    act = rec["act"]
    state = np.hstack([act[:, ASSOC], act[:, PFC], act[:, MOTOR]])
    feats = {
        "frame": rec["frame"],
        "two_frame": np.hstack([rec["frame"], rec["prev"]]),
        "state": state,
        "assoc": act[:, ASSOC],
        "pfc": act[:, PFC],
        "frame_state": np.hstack([rec["frame"], state]),
        "frame_trace": np.hstack([rec["frame"], rec["trace"]]),
    }
    eps = np.unique(rec["ep"])
    # spread episodes over folds evenly (episode lengths vary; interleave by index)
    groups = np.searchsorted(eps, rec["ep"])
    targets = {
        "dir4": ((dx + 1) * 3 + (dy + 1), keep),
        "dx": (dx, keep),
        "dy": (dy, keep),
        "ball_dir": (rec["dir"], np.ones(len(dx), bool)),
    }
    out = {"moments": int(len(dx)), "episodes": int(len(eps)), "dropped_dir4": int((~keep).sum()),
           "agree": round(float(rec["agree"].mean()), 4), "probes": {}}
    for tname, (y, m) in targets.items():
        out["probes"][tname] = {fname: probe(X[m], y[m], groups[m]) for fname, X in feats.items()}
    return out


# -- life -------------------------------------------------------------------------------------

def life(job):
    arm, seed, blocks = job
    breakout, run = _classic()
    player = run.make(arm, seed, dict(run.CADENCE), run.MLP_LR)
    env = breakout.Breakout(seed=seed)
    t0 = time.time()
    out = {"arm": arm, "seed": seed, "score": [], "checkpoints": {}}
    raw = {}

    def checkpoint(block):
        rec = collect(player, seed, block)
        raw.update({f"b{block}_{k}": v for k, v in rec.items() if k in ("bx", "by", "lx", "ly", "dir", "ep", "act", "trace")})
        res = analyse(rec)
        les = lesion_scores(player, seed, block)
        res["lesion"] = {c: float(np.mean(v)) for c, v in les.items()}
        res["lesion_raw"] = les
        out["checkpoints"][str(block)] = res
        pr = res["probes"]["dir4"]
        print(arm, seed, "block", block, "dir4 frame", pr["frame"]["acc"], "state", pr["state"]["acc"],
              "frame_trace", pr["frame_trace"]["acc"], "maj", pr["frame"]["majority"],
              "lesion", {c: round(v, 2) for c, v in res["lesion"].items()},
              f"{time.time() - t0:.0f}s", flush=True)

    k = 0
    player.new_episode()
    if 0 in CHECKPOINTS:
        checkpoint(0)
    for block in range(blocks):
        # identical to classic/run.py life()
        for _ in range(run.TRAIN):
            obs, prev = env.observe(), env.previous()
            label = env.coach()
            a = player.train(obs, prev, label)
            _, done = env.step(a); k += 1
            if done or k >= run.CAP:
                env.reset(); player.new_episode(); k = 0
        env.reset(); player.new_episode(); k = 0
        scores, _, _, _ = run.test(player, seed, block, False)
        player.new_episode()
        out["score"].append(scores)
        if block + 1 in CHECKPOINTS:
            checkpoint(block + 1)
    out["seconds"] = round(time.time() - t0, 1)
    np.savez_compressed(OUT / f"raw-{arm}-{seed}.npz", **{k: v.astype(np.float32) for k, v in raw.items()})
    return out


def summarize():
    from scipy.stats import spearmanr
    rows = {}
    for f in sorted(OUT.glob("*-*.json")):
        if f.name.startswith(("summary", "meta")):
            continue
        r = json.loads(f.read_text())
        rows.setdefault(r["arm"], []).append(r)
    long = CLASSIC / "runs" / "long10x"
    summ = {"arms": {}}
    for arm, rs in rows.items():
        rs.sort(key=lambda r: r["seed"])
        a = {"seeds": [r["seed"] for r in rs], "reproduces_long10x": {}, "table": {}}
        for r in rs:
            ref = long / f"{arm}-{r['seed']}.json"
            if ref.exists():
                old = json.loads(ref.read_text())["score"][: len(r["score"])]
                a["reproduces_long10x"][r["seed"]] = old == r["score"]
        for b in map(str, CHECKPOINTS):
            if not all(b in r["checkpoints"] for r in rs):
                continue
            t = {}
            for tname in ("dir4", "dx", "dy", "ball_dir"):
                for fname in rs[0]["checkpoints"][b]["probes"][tname]:
                    accs = [r["checkpoints"][b]["probes"][tname][fname]["acc"] for r in rs]
                    t[f"{tname}/{fname}"] = accs
                t[f"{tname}/majority"] = [r["checkpoints"][b]["probes"][tname]["frame"]["majority"] for r in rs]
            for c in ("intact", "wm0", "cold", "both"):
                t[f"lesion/{c}"] = [round(r["checkpoints"][b]["lesion"][c], 2) for r in rs]
            t["agree"] = [r["checkpoints"][b]["agree"] for r in rs]
            t["moments"] = [r["checkpoints"][b]["moments"] for r in rs]
            a["table"][b] = t
        acc, bricks = [], []
        for r in rs:
            for b in ("10", "30", "75", "150"):
                if b in r["checkpoints"]:
                    acc.append(r["checkpoints"][b]["probes"]["dir4"]["state"]["acc"])
                    bricks.append(r["checkpoints"][b]["lesion"]["intact"])
        if len(acc) > 2:
            rho, p = spearmanr(acc, bricks)
            a["spearman_state_dir4_vs_bricks"] = {"rho": round(float(rho), 3), "p": round(float(p), 4),
                                                  "n": len(acc)}
        a["block_test_last10"] = {r["seed"]: {"30": round(float(np.mean(r["score"][20:30])), 2),
                                             "150": round(float(np.mean(r["score"][140:150])), 2)}
                                  for r in rs if len(r["score"]) >= 150}
        summ["arms"][arm] = a
    _write(OUT / "summary.json", json.dumps(summ, indent=1) + "\n")
    for arm, a in summ["arms"].items():
        print("==", arm, "seeds", a["seeds"], "reproduces", a["reproduces_long10x"])
        for b, t in a["table"].items():
            m = lambda k: np.round(np.mean(t[k]), 3)
            print(f" block {b:>3}: dir4 maj {m('dir4/majority')} frame {m('dir4/frame')} two {m('dir4/two_frame')}"
                  f" state {m('dir4/state')} pfc {m('dir4/pfc')} f+state {m('dir4/frame_state')}"
                  f" f+trace {m('dir4/frame_trace')} | bricks intact {m('lesion/intact')} wm0 {m('lesion/wm0')}"
                  f" cold {m('lesion/cold')} both {m('lesion/both')}")
        print(" spearman", a.get("spearman_state_dir4_vs_bricks"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="norecall")
    ap.add_argument("--seeds", default="10-13")
    ap.add_argument("--blocks", type=int, default=150)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.summarize:
        summarize(); return
    lo, hi = (int(x) for x in a.seeds.split("-"))
    import cadence as cd
    meta = {"cadence": cd.__version__, "python": platform.python_version(), "blocks": a.blocks,
            "checkpoints": CHECKPOINTS, "probe_moments": PROBE_MOMENTS, "lesion_eps": LESION_EPS,
            "l2": L2, "folds": FOLDS, "arms": a.arms.split(","), "seeds": [lo, hi]}
    _write(OUT / "meta.json", json.dumps(meta, indent=1) + "\n")
    jobs = [(arm, s, a.blocks) for arm in a.arms.split(",") for s in range(lo, hi + 1)]
    with Pool(a.procs) as pool:
        for r in pool.imap_unordered(life, jobs):
            _write(OUT / f"{r['arm']}-{r['seed']}.json", json.dumps(r, separators=(",", ":")) + "\n")
            print("done", r["arm"], r["seed"], f"{r['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
