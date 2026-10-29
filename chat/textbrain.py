"""Text in and out of a Cadence brain, one character per moment.

Input: the observation at moment t is the one-hot code of the last ``window`` characters
(the current one last), ``window * V`` values. Optional opt-in heard history
(``history_decays``) adds fading copies of those observations at several timescales.

Output: one action slot with V choices. The action issued at moment t is the brain's guess at
character t+1.

Learning signal: supervised teaching. ``Brain.step(x_t, teacher=[c_{t+1}])`` teaches the true
next character as the label of the present observation (the composed teaching learner, a
free and a nudged settle). The reward learner's rates are zero: the text pays nothing.

Reading: after a greedy ``act`` the motor policy is ``basal_ganglia.probabilities(state, T)``.
Bits per character use a temperature T picked on the validation text (each learner gets its
own calibration; accuracy needs none).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import cadence as cd
import corpus as C

NEWLINE = C.INDEX["\n"]


def encoder(window: int):
    def enc(history: list[int]) -> np.ndarray:
        x = np.zeros((1, window * C.V))
        tail = history[-window:]
        off = window - len(tail)
        for j, c in enumerate(tail):
            x[0, (off + j) * C.V + c] = 1.0
        return x
    return enc


def compose(cfg: dict, seed: int = 0) -> cd.Brain:
    genes = dict(actor_eta=0.0, actor_eta_bias=0.0, actor_eta_critic=0.0)
    genes.update(cfg.get("genes", {}))
    return cd.Brain.compose(cfg["window"] * C.V, C.V, seed=seed, **genes)


def probs(brain: cd.Brain, temperature: float) -> np.ndarray:
    bg = brain.basal_ganglia
    p = np.asarray(bg.probabilities(bg.state, temperature=temperature), dtype=float)
    return p.reshape(-1)[: C.V]


TEMPS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0)


def score_text(brain: cd.Brain, ids: list[int], window: int, warm: list[int]) -> dict:
    """Teacher-forced pass, greedy, no learning: next-char accuracy and bits per char at
    each temperature. ``warm`` is fed first, unscored."""
    enc = encoder(window)
    brain.reset()
    hist: list[int] = []
    for c in warm:
        hist.append(c)
        brain.act(enc(hist), greedy=True)
    hits = 0
    nll = np.zeros(len(TEMPS))
    n = 0
    reply_hits = reply_n = 0
    in_reply = False
    for t in range(len(ids) - 1):
        hist.append(ids[t])
        a = int(brain.act(enc(hist), greedy=True)[0])
        y = ids[t + 1]
        hits += a == y
        for i, T in enumerate(TEMPS):
            nll[i] -= np.log2(max(probs(brain, T)[y], 1e-12))
        n += 1
        # characters of bot replies (after "bot: ")
        if in_reply:
            reply_hits += a == y
            reply_n += 1
        if ids[t] == NEWLINE:
            in_reply = False
        if ids[t] == C.INDEX[" "] and C.decode(hist[-5:]) == "bot: ":
            in_reply = True
    return {"acc": hits / n, "bpc": {str(T): v / n for T, v in zip(TEMPS, nll)},
            "reply_acc": reply_hits / max(reply_n, 1), "chars": n}


def say(brain: cd.Brain, hist: list[int], prompt: str, window: int, max_len: int = 60,
        temperature: float | None = None, rng: np.random.Generator | None = None) -> str:
    """Read "you: <prompt>\nbot: " one char per moment, then let the brain write: each issued
    action is the next char, which it reads at the next moment, until it issues a newline.
    The brain then reads that newline, so the stream is ready for the next prompt."""
    enc = encoder(window)

    def step(c: int) -> int:
        hist.append(c)
        a = int(brain.act(enc(hist), greedy=True)[0])
        if temperature is not None:
            p = probs(brain, temperature)
            a = int((rng or np.random.default_rng()).choice(C.V, p=p / p.sum()))
        return a

    a = NEWLINE
    for c in C.encode(f"you: {prompt}\nbot: "):
        a = step(c)
    out = []
    while a != NEWLINE and len(out) < max_len:
        out.append(C.ALPHABET[a])
        a = step(a)
    step(NEWLINE)
    if len(hist) > 256:
        del hist[:-64]
    return "".join(out)


def converse(brain: cd.Brain, prompts: list[str], window: int) -> list[str]:
    """One continuing conversation: each prompt, then the brain's own reply, which it keeps
    hearing as the conversation goes on."""
    hist: list[int] = []
    return [say(brain, hist, p, window) for p in prompts]


def reply_eval(brain: cd.Brain, exs, window: int) -> dict:
    brain.reset()
    prompts = [p for p, _, _, _ in exs]
    replies = converse(brain, prompts, window)
    ok = [C.valid_reply(p, r) for p, r in zip(prompts, replies)]
    by = {}
    for (_, _, _, name), v in zip(exs, ok):
        by.setdefault(name, []).append(bool(v))
    return {"valid": float(np.mean([bool(v) for v in ok])),
            "by_intent": {k: float(np.mean(v)) for k, v in by.items()},
            "samples": [[p, r] for p, r in list(zip(prompts, replies))[:12]]}


def evaluate(brain: cd.Brain, window: int, val_ids, warm, val_exs, novel_exs) -> dict:
    s = score_text(brain, val_ids, window, warm)
    r = reply_eval(brain, val_exs, window)
    nv = reply_eval(brain, novel_exs, window)
    return {"text": s, "reply": r, "novel": {"valid": nv["valid"], "by_intent": nv["by_intent"],
                                             "samples": nv["samples"][:6]}}


def train(cfg: dict, out: Path, chars: int, probe_every: int, seed: int = 0,
          resume: Path | None = None, start_char: int = 0, log=print) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tr, va, te = C.splits()
    text_ids = C.encode(C.render(tr))
    val_exs = va[:120]
    val_ids = C.encode(C.render(val_exs))
    warm = C.encode(C.render(va[200:202]))
    novel = C.novel_exchanges(40)
    window = cfg["window"]
    enc = encoder(window)
    if resume is not None:
        brain = cd.Brain.load(resume)
    else:
        brain = compose(cfg, seed)
    (out / "cfg.json").write_text(json.dumps(cfg), encoding="utf-8")
    log(json.dumps({"event": "start", "cfg": cfg, "version": cd.__version__,
                    "neurons": brain.connectome.n, "synapses": brain.connectome.synapses,
                    "parameters": brain.parameters(), "start_char": start_char}))
    hist: list[int] = text_ids[max(0, (start_char % len(text_ids)) - window):start_char % len(text_ids)]
    t0 = time.monotonic()
    lessons = hits = 0
    since = 0
    pos = start_char
    end = start_char + chars
    while pos < end:
        i = pos % len(text_ids)
        c = text_ids[i]
        y = text_ids[(i + 1) % len(text_ids)]
        hist.append(c)
        if len(hist) > 64:
            del hist[:-16]
        a = int(brain.step(enc(hist), teacher=[y])[0])
        hits += a == y
        lessons += int(brain.last_learning.get("demonstration_accepted_presentations", 1))
        since += 1
        pos += 1
        if pos % probe_every == 0 or pos == end:
            path = brain.save(out / "brain.npz")
            probe = cd.Brain.load(path)
            ev = evaluate(probe, window, val_ids, warm, val_exs, novel)
            best_T = min(ev["text"]["bpc"], key=ev["text"]["bpc"].get)
            log(json.dumps({"event": "probe", "chars": pos, "seconds": round(time.monotonic() - t0, 1),
                            "train_acc": round(hits / since, 3), "lesson_share": round(lessons / since, 3),
                            "val_acc": round(ev["text"]["acc"], 4), "val_reply_acc": round(ev["text"]["reply_acc"], 4),
                            "val_bpc": round(ev["text"]["bpc"][best_T], 3), "T": best_T,
                            "valid_reply": round(ev["reply"]["valid"], 3), "novel_valid": round(ev["novel"]["valid"], 3)}))
            (out / f"eval-{pos}.json").write_text(json.dumps(ev, indent=1), encoding="utf-8")
            hits = lessons = since = 0
