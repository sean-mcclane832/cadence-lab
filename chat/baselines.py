"""Baselines on the same corpus and the same scoring as the Cadence brain.

    python baselines.py ngram
    python baselines.py mlp --chars 300000
    python baselines.py gru --chars 300000

Each model is a stream predictor: ``reset()``, then ``read(c)`` returns the probability of
every next character. Accuracy is argmax; bits per char use the model's own probabilities
(a softmax temperature picked on validation, as for Cadence). Replies are greedy, the model
reading its own output, exactly like ``textbrain.say``.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

import corpus as C  # noqa: E402

NL = C.INDEX["\n"]
TEMPS = (0.5, 0.8, 1.0, 1.2, 1.5)


class NGram:
    """Interpolated Witten-Bell character n-gram."""

    def __init__(self, order: int):
        self.n = order
        self.counts = [defaultdict(lambda: np.zeros(C.V)) for _ in range(order)]

    def fit(self, ids):
        for t in range(len(ids)):
            for k in range(self.n):
                if t - k < 0:
                    break
                ctx = tuple(ids[t - k:t])
                self.counts[k][ctx][ids[t]] += 1

    def reset(self):
        self.h = []

    def read(self, c, T=1.0):
        self.h.append(c)
        p = np.full(C.V, 1.0 / C.V)
        for k in range(self.n):
            if k > len(self.h):
                break
            ctx = tuple(self.h[len(self.h) - k:]) if k else ()
            cnt = self.counts[k].get(ctx)
            if cnt is None:
                break
            tot = cnt.sum()
            types = (cnt > 0).sum()
            lam = tot / (tot + types)
            p = lam * cnt / tot + (1 - lam) * p
        if T != 1.0:
            p = p ** (1 / T)
            p /= p.sum()
        return p


class TorchModel:
    def __init__(self, kind: str, hidden: int, window: int = 3, seed: int = 0):
        import torch
        import torch.nn as nn
        torch.manual_seed(seed)
        torch.set_num_threads(1)
        self.torch, self.kind, self.window = torch, kind, window
        if kind == "gru":
            self.emb = nn.Embedding(C.V, C.V)
            self.emb.weight.data = torch.eye(C.V)
            self.emb.weight.requires_grad_(False)
            self.rnn = nn.GRU(C.V, hidden, batch_first=True)
            self.out = nn.Linear(hidden, C.V)
            self.mods = nn.ModuleList([self.rnn, self.out])
        else:
            self.net = nn.Sequential(nn.Linear(window * C.V, hidden), nn.Tanh(), nn.Linear(hidden, C.V))
            self.mods = nn.ModuleList([self.net])
        self.params = sum(p.numel() for p in self.mods.parameters())

    def _win(self, ids, t):
        x = np.zeros(self.window * C.V, dtype=np.float32)
        for j in range(self.window):
            i = t - (self.window - 1 - j)
            if i >= 0:
                x[j * C.V + ids[i]] = 1
        return x

    def fit(self, ids, chars, log, seq=64, batch=16, lr=3e-3):
        torch = self.torch
        opt = torch.optim.Adam(self.mods.parameters(), lr=lr)
        ids_t = torch.tensor(ids)
        seen = 0
        rng = np.random.default_rng(0)
        while seen < chars:
            starts = rng.integers(0, len(ids) - seq - 1, size=batch)
            if self.kind == "gru":
                x = torch.stack([ids_t[s:s + seq] for s in starts])
                y = torch.stack([ids_t[s + 1:s + seq + 1] for s in starts])
                h, _ = self.rnn(torch.nn.functional.one_hot(x, C.V).float())
                logits = self.out(h)
            else:
                pos = np.concatenate([np.arange(s, s + seq) for s in starts])
                x = torch.tensor(np.stack([self._win(ids, t) for t in pos]))
                y = ids_t[pos + 1]
                logits = self.net(x)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, C.V), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            seen += batch * seq
            if seen % (batch * seq * 200) == 0:
                log(json.dumps({"seen": seen, "loss_bits": round(loss.item() / np.log(2), 3)}))

    def reset(self):
        self.h = []
        self.state = None

    def read(self, c, T=1.0):
        torch = self.torch
        self.h.append(c)
        with torch.no_grad():
            if self.kind == "gru":
                x = torch.nn.functional.one_hot(torch.tensor([[c]]), C.V).float()
                o, self.state = self.rnn(x, self.state)
                logits = self.out(o[0, -1])
            else:
                logits = self.net(torch.tensor(self._win(self.h, len(self.h) - 1)))
            return torch.softmax(logits / T, -1).numpy().astype(float)


def score_text(m, ids, warm):
    m.reset()
    for c in warm:
        m.read(c)
    hits, n = 0, 0
    nll = np.zeros(len(TEMPS))
    for t in range(len(ids) - 1):
        ps = [m_read for m_read in [None]]
        p1 = m.read(ids[t], 1.0)
        y = ids[t + 1]
        hits += int(np.argmax(p1) == y)
        lp = np.log(np.maximum(p1, 1e-300))
        for i, T in enumerate(TEMPS):
            q = np.exp((lp - lp.max()) / T)
            q /= q.sum()
            nll[i] -= np.log2(max(q[y], 1e-12))
        n += 1
    return {"acc": hits / n, "bpc": {str(T): v / n for T, v in zip(TEMPS, nll)}}


def say(m, prompt, max_len=60):
    p = None
    for c in C.encode(f"you: {prompt}\nbot: "):
        p = m.read(c)
    out = []
    a = int(np.argmax(p))
    while a != NL and len(out) < max_len:
        out.append(C.ALPHABET[a])
        a = int(np.argmax(m.read(a)))
    m.read(NL)
    return "".join(out)


def reply_eval(m, exs):
    m.reset()
    rs = [say(m, p) for p, _, _, _ in exs]
    ok = [bool(C.valid_reply(p, r)) for (p, _, _, _), r in zip(exs, rs)]
    by = {}
    for (_, _, _, name), v in zip(exs, ok):
        by.setdefault(name, []).append(v)
    return {"valid": float(np.mean(ok)), "by_intent": {k: float(np.mean(v)) for k, v in by.items()},
            "samples": [[p, r] for (p, _, _, _), r in list(zip(exs, rs))[:12]]}


def full_eval(m):
    tr, va, te = C.splits()
    val = score_text(m, C.encode(C.render(va[:120])), C.encode(C.render(va[200:202])))
    T = min(val["bpc"], key=val["bpc"].get)
    test = score_text(m, C.encode(C.render(te)), C.encode(C.render(va[200:202])))
    return {"val": val, "T": T, "test_acc": test["acc"], "test_bpc": test["bpc"][T],
            "test_reply": reply_eval(m, te), "novel": reply_eval(m, C.novel_exchanges(40))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["ngram", "mlp", "gru"])
    ap.add_argument("--chars", type=int, default=300_000)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--out", type=Path, default=Path("runs/baselines"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    tr, va, te = C.splits()
    ids = C.encode(C.render(tr))

    def log(s):
        print(s, flush=True)

    t0 = time.time()
    if a.kind == "ngram":
        res = {}
        for order in (2, 3, 4, 6, 8, 10, 12):
            m = NGram(order)
            m.fit(ids)
            r = full_eval(m)
            res[order] = r
            log(json.dumps({"order": order, "val_acc": round(r["val"]["acc"], 4), "test_acc": round(r["test_acc"], 4),
                            "test_bpc": round(r["test_bpc"], 3), "valid_reply": r["test_reply"]["valid"],
                            "novel": r["novel"]["valid"]}))
        best = max(res, key=lambda k: res[k]["val"]["acc"])
        out = {"kind": "ngram", "orders": {str(k): {kk: v[kk] for kk in ("test_acc", "test_bpc")} | {"valid": v["test_reply"]["valid"], "novel": v["novel"]["valid"], "val_acc": v["val"]["acc"]} for k, v in res.items()},
               "best_order": best, "best": res[best]}
    else:
        m = TorchModel(a.kind, a.hidden)
        log(json.dumps({"kind": a.kind, "params": m.params}))
        m.fit(ids, a.chars, log)
        r = full_eval(m)
        out = {"kind": a.kind, "params": m.params, "chars": a.chars, **r}
    out["seconds"] = round(time.time() - t0, 1)
    name = a.kind if a.kind == "ngram" else f"{a.kind}-h{a.hidden}-{a.chars}"
    (a.out / f"{name}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    b = out.get("best", out)
    log(json.dumps({"done": name, "test_acc": round(b["test_acc"], 4), "test_bpc": round(b["test_bpc"], 3),
                    "valid_reply": b["test_reply"]["valid"], "novel": b["novel"]["valid"]}))


if __name__ == "__main__":
    main()
