"""A Cadence brain that reads and writes one word token per moment.

Same recipe as textbrain.py (chars): observation = one-hot of the last ``window`` tokens
(+ opt-in heard history), one action slot over the vocabulary, the issued action is the
next token, taught by ``Brain.step(x_t, teacher=[w_{t+1}])`` with reward rates at zero.

    python wordbrain.py --cfg w6 --tokens 200000 --probe 20000 --out runs/words/w6
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

for k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")

import numpy as np  # noqa: E402

import cadence as cd  # noqa: E402
import corpus as C  # noqa: E402
import words as W  # noqa: E402

H3 = {"history_decays": (0.2, 0.8, 0.95)}
NORM = {"learning_surprise": True, "teaching": "normalized"}
CONFIGS = {
    "w1": {"window": 1, "genes": {**NORM, **H3}},
    "w4": {"window": 4, "genes": {**NORM, **H3}},
    "w6": {"window": 6, "genes": {**NORM, **H3}},
    "w8": {"window": 8, "genes": {**NORM, **H3}},
    "w6_m128": {"window": 6, "genes": {**NORM, **H3, "modules": (128,)}},
    "w8_m128": {"window": 8, "genes": {**NORM, **H3, "modules": (128,)}},
    "w8_nohist": {"window": 8, "genes": {**NORM}, "loop": "step"},
    # supervised-only loop (lesson, then greedy act) and no associative store: the text
    # pays nothing, so the store only ever recorded zero outcomes; 2.6x faster
    "s8": {"window": 8, "genes": {**NORM, "episodic": False}},
    "s8_m128": {"window": 8, "genes": {**NORM, "episodic": False, "modules": (128,)}},
    "s8_b16": {"window": 8, "streams": 16, "genes": {**NORM, "episodic": False}},
    "s8_b32": {"window": 8, "streams": 32, "genes": {**NORM, "episodic": False}},
    "s8_b16_m128": {"window": 8, "streams": 16, "genes": {**NORM, "episodic": False, "modules": (128,)}},
    "s12_b16": {"window": 12, "streams": 16, "genes": {**NORM, "episodic": False}},
    "s8_b16_e1": {"window": 8, "streams": 16, "eta": 0.001, "genes": {**NORM, "episodic": False}},
    "s8_b16_e03": {"window": 8, "streams": 16, "eta": 0.0003, "genes": {**NORM, "episodic": False}},
    "s8_b16_e1_cap2": {"window": 8, "streams": 16, "eta": 0.001, "scale_cap": 2.0,
                       "genes": {**NORM, "episodic": False}},
    "s8_b16_e03_r1k": {"window": 8, "streams": 16, "eta": 0.0003, "reset_every": 1024,
                       "genes": {**NORM, "episodic": False}},
    "main_cold": {"window": 8, "streams": 16, "eta": 0.0003, "reset_every": 1024, "cold": True,
                  "genes": {**NORM, "episodic": False}},
    "main_cold_e1": {"window": 8, "streams": 16, "eta": 0.001, "reset_every": 1024, "cold": True,
                     "genes": {**NORM, "episodic": False}},
    "cold_m128": {"window": 8, "streams": 16, "eta": 0.0003, "reset_every": 1024, "cold": True,
                  "genes": {**NORM, "episodic": False, "modules": (128,)}},
    "cold_e03_scratch": {"window": 8, "streams": 16, "eta": 0.0003, "reset_every": 1024, "cold": True,
                         "genes": {**NORM, "episodic": False}},
    "main_cold_anneal": {"window": 8, "streams": 16, "eta": 0.0001, "reset_every": 1024, "cold": True,
                         "genes": {**NORM, "episodic": False}},
    # upstream 0.91 copy pathway: recall memory with direct read -> motor synapses that start
    # as the identity from each token's current-position input coordinate to its action
    "copy8": {"window": 8, "streams": 16, "eta": 0.0003, "reset_every": 1024, "cold": True,
              "genes": {**NORM, "episodic": False, "recall": True, "recall_copy": 2.0,
                        "recall_tokens": [7 * W.V + j for j in range(W.V)]}},
    "main_nudge32": {"window": 8, "streams": 16, "eta": 0.0001, "reset_every": 1024, "cold": True,
                     "retune": {"learning_nudged_steps": 32}, "genes": {**NORM, "episodic": False}},
    "s10": {"window": 10, "genes": {**NORM, "episodic": False}},
    "w6_nohist": {"window": 6, "genes": {**NORM}, "loop": "step"},
    "w6_nohist_m128": {"window": 6, "genes": {**NORM, "modules": (128,)}},
    "w8_nohist_m128": {"window": 8, "genes": {**NORM, "modules": (128,)}},
}
TEMPS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2)


def cold(brain) -> None:
    """With ``brain.chat_cold`` set, the next settle starts from rest instead of from the
    stream's last settled state (the warm start). The working trace and every memory are
    kept. This uses a private attribute (``basal_ganglia._free``), the same lever the lab's
    settle probe used for its "cold" lesion: a warm-started stream can fall into a bad
    settled state and stay there (see PROGRESS.md, stage 3)."""
    if getattr(brain, "chat_cold", False):
        brain.basal_ganglia._free = None


def encoder(window: int):
    def enc(hist: list[int]) -> np.ndarray:
        x = np.zeros((1, window * W.V))
        tail = hist[-window:]
        off = window - len(tail)
        for j, c in enumerate(tail):
            x[0, (off + j) * W.V + c] = 1.0
        return x
    return enc


def compose(cfg: dict, seed: int = 0) -> cd.Brain:
    genes = dict(actor_eta=0.0, actor_eta_bias=0.0, actor_eta_critic=0.0)
    genes.update(cfg.get("genes", {}))
    return cd.Brain.compose(cfg["window"] * W.V, W.V, seed=seed, **genes)


def probs(brain: cd.Brain, T: float) -> np.ndarray:
    bg = brain.basal_ganglia
    return np.asarray(bg.probabilities(bg.state, temperature=T), dtype=float).reshape(-1)[: W.V]


def score_tokens(brain, ids, window, warm) -> dict:
    enc = encoder(window)
    brain.reset()
    hist: list[int] = []
    for c in warm:
        hist.append(c)
        cold(brain)
        brain.act(enc(hist), greedy=True)
    hits = n = rh = rn = 0
    nll = np.zeros(len(TEMPS))
    in_reply = False
    for t in range(len(ids) - 1):
        hist.append(ids[t])
        cold(brain)
        a = int(brain.act(enc(hist), greedy=True)[0])
        y = ids[t + 1]
        hits += a == y
        for i, T in enumerate(TEMPS):
            nll[i] -= np.log2(max(probs(brain, T)[y], 1e-12))
        n += 1
        if ids[t] == W.BOT_ID:
            in_reply = True
        if in_reply:
            rh += a == y
            rn += 1
        if y == W.EOL_ID:
            in_reply = False
    return {"acc": hits / n, "bpt": {str(T): v / n for T, v in zip(TEMPS, nll)},
            "reply_acc": rh / max(rn, 1), "tokens": n}


def say(brain, hist, prompt, window, max_len=20, temperature=None, rng=None) -> str:
    enc = encoder(window)

    def step(c: int) -> int:
        hist.append(c)
        cold(brain)
        a = int(brain.act(enc(hist), greedy=True)[0])
        if temperature is not None:
            p = probs(brain, temperature)
            a = int((rng or np.random.default_rng()).choice(W.V, p=p / p.sum()))
        return a

    a = W.EOL_ID
    toks = [W.INDEX[t] for t in W.tokenize(f"you: {prompt}")] + [W.BOT_ID]
    for c in toks:
        a = step(c)
    out = []
    while a != W.EOL_ID and len(out) < max_len:
        out.append(W.VOCAB[a])
        a = step(a)
    step(W.EOL_ID)
    if len(hist) > 256:
        del hist[:-64]
    return W.detok(out)


def reply_eval(brain, exs, window) -> dict:
    brain.reset()
    hist: list[int] = []
    rs = [say(brain, hist, p, window) for p, _, _, _ in exs]
    ok = [bool(C.valid_reply(p, r)) for (p, _, _, _), r in zip(exs, rs)]
    by: dict = {}
    for (_, _, _, name), v in zip(exs, ok):
        by.setdefault(name, []).append(v)
    return {"valid": float(np.mean(ok)), "by_intent": {k: float(np.mean(v)) for k, v in by.items()},
            "samples": [[p, r] for (p, _, _, _), r in list(zip(exs, rs))[:12]]}


def evaluate(brain, window, split="val") -> dict:
    tr, va, te = C.splits()
    exs = va[:120] if split == "val" else te
    s = score_tokens(brain, W.encode(C.render(exs)), window, W.encode(C.render(va[200:202])))
    r = reply_eval(brain, exs, window)
    nv = reply_eval(brain, C.novel_exchanges(40), window)
    return {"split": split, "text": s, "reply": r, "novel": nv}


def train(cfg, out: Path, tokens: int, probe_every: int, seed=0, resume=None, start=0, log=print):
    out.mkdir(parents=True, exist_ok=True)
    (out / "cfg.json").write_text(json.dumps(cfg), encoding="utf-8", newline="\n")
    tr, _, _ = C.splits()
    ids = W.encode(C.render(tr))
    window = cfg["window"]
    enc = encoder(window)
    brain = cd.Brain.load(resume) if resume else compose(cfg, seed)
    if cfg.get("retune"):
        brain.retune(**cfg["retune"])
    if "eta" in cfg or "scale_cap" in cfg:
        # smaller teaching steps than the normalized point's 0.003 / 0.0003 (compose does
        # not take eta while teaching= sets it, so the learner's config is replaced)
        import dataclasses
        c = brain.learner.config
        brain.learner.config = dataclasses.replace(
            c, eta=cfg.get("eta", c.eta), eta_bias=cfg.get("eta", c.eta) / 10,
            scale_cap=cfg.get("scale_cap", c.scale_cap))
        c = brain.learner.config
        log(json.dumps({"event": "learner", "eta": c.eta, "eta_bias": c.eta_bias, "scale_cap": c.scale_cap}))
    brain.chat_cold = bool(cfg.get("cold"))
    log(json.dumps({"event": "start", "cfg": cfg, "version": cd.__version__, "vocab": W.V,
                    "neurons": brain.connectome.n, "synapses": brain.connectome.synapses,
                    "parameters": brain.parameters(), "start": start}))
    # B parallel streams through the text, evenly spaced; one lesson teaches all B rows
    B = int(cfg.get("streams", 1))
    brain.reset()
    hists: list[list[int]] = [[] for _ in range(B)]
    t0 = time.monotonic()
    hits = lessons = since = refusals = 0
    pos = start
    t = start // B
    while pos < start + tokens:
        xs, ys = [], []
        for k in range(B):
            i = (k * (len(ids) // B) + t) % len(ids)
            h = hists[k]
            h.append(ids[i])
            if len(h) > 64:
                del h[:-16]
            xs.append(enc(h)[0])
            ys.append(ids[(i + 1) % len(ids)])
        x, y = np.array(xs), np.array(ys)
        t += 1
        if cfg.get("reset_every") and t % cfg["reset_every"] == 0:
            brain.reset()  # clear the streams' traces; the weights carry on
        if cfg.get("loop") == "step":
            a = brain.step(x, teacher=y)
        else:
            # upstream's supervised-only stream: teach the present observation, then a
            # greedy act advances the stream (no reward eligibility, no outcome records)
            cold(brain)
            try:
                brain.lesson(x, y)
                cold(brain)
                a = brain.act(x, greedy=True)
            except RuntimeError as exc:
                # a refused settle (budget exhausted) changes nothing; count it, clear the
                # streams and go on with the next moment
                refusals += 1
                log(json.dumps({"event": "refusal", "tokens": pos, "error": str(exc)[:160]}))
                brain.reset()
                a = np.full(B, -1)
        hits += int((np.asarray(a).reshape(-1) == y).sum())
        lessons += int(brain.last_learning.get("demonstration_accepted_presentations", B))
        since += B
        pos += B
        if pos % probe_every < B or pos >= start + tokens:
            path = brain.save(out / "brain.npz")
            probe = cd.Brain.load(path)
            probe.chat_cold = brain.chat_cold
            try:
                ev = evaluate(probe, window)
            except RuntimeError as exc:
                log(json.dumps({"event": "probe_refused", "tokens": pos, "error": str(exc)[:160]}))
                continue
            T = min(ev["text"]["bpt"], key=ev["text"]["bpt"].get)
            log(json.dumps({"event": "probe", "tokens": pos, "epochs": round(pos / len(ids), 2),
                            "seconds": round(time.monotonic() - t0, 1),
                            "train_acc": round(hits / since, 3), "lesson_share": round(lessons / since, 3),
                            "val_acc": round(ev["text"]["acc"], 4), "val_reply_acc": round(ev["text"]["reply_acc"], 4),
                            "val_bpt": round(ev["text"]["bpt"][T], 3), "T": T,
                            "valid_reply": round(ev["reply"]["valid"], 3), "novel_valid": round(ev["novel"]["valid"], 3),
                            "refusals": refusals}))
            (out / f"eval-{pos}.json").write_text(json.dumps(ev, indent=1), encoding="utf-8", newline="\n")
            if pos % (probe_every * 5) < B:
                brain.save(out / f"brain-{pos}.npz")
            hits = lessons = since = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--tokens", type=int, default=100_000)
    ap.add_argument("--probe", type=int, default=20_000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--start", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    def log(line):
        print(line, flush=True)
        with open(a.out / "log.jsonl", "a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")

    train(dict(CONFIGS[a.cfg], name=a.cfg), a.out, a.tokens, a.probe, resume=a.resume, start=a.start, log=log)


if __name__ == "__main__":
    main()
