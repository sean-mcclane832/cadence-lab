"""Extinction rules for issue #210, applied as patches of this process (no library change).

`apply(rule, theta)` patches `cadence.memories.EpisodicRecords.outcome` (and wraps
`Brain.learn` so the outcome can see the next observation's place key). Rules:

  off      cadence 0.89.0 as released: a zero outcome mid-attempt writes nothing.
  owner    the owner's `memory_extinction` gene as described on #210 (20 Oct): a learned
           zero outcome is written when (1) the vote held for the chosen action is >= theta
           and (2) the decision left its place key unchanged (next key == decision key).
  direct   ours: a learned zero outcome is written when (1) the vote held for the chosen
           action is >= theta and (2) this action has been paid *directly* in this place:
           a known outcome of the action itself with reward > 0 (not replay, not a
           demonstration). A replayed corridor credit or a taught note never qualifies,
           so only a payment that stopped is extinguished.
  lib      the library gene `memory_extinction` (P:/projects/cadence-210fix, the `direct`
           rule kept in an associative store); needs PYTHONPATH on that copy's src.
  libctx   (v2, 2026-10-28) the library gene `lib`, plus a context flag: the zero is written
           only if this action was also paid directly after a similar history, i.e. the
           decision's context key (the temporal records' key, which carries the recent
           observations and the efference copy) has cosine similarity >= CTX_MATCH to a context
           key at which a known outcome of this action paid. Aimed at the hidden chain, where
           the observation is the same at every stage but the history is not. A prototype in
           this process (a list of paid context keys per action), not a library change.
  history  ours, narrower: as `direct`, but judged and written in the context records only:
           the context record of this decision (keyed by its temporal history) must hold
           >= theta, and the zero is written there alone, the place record left as it is.
           A paid action that pays only after some histories (a hidden chain) is then
           extinguished only for the histories in which it did not pay.

The direct table is keyed by the exact place key (bytes) and the action; it suits the
fixed, deterministic codes of these chambers (one-hot observations, default sparse code).
The library version in P:/projects/cadence-210fix keeps it in an associative store.
A written zero is marked with salience 1e-9, as in the owner's candidate: it becomes a known
outcome without speeding consolidation and without changing the reward (so replay is not
bounded by it).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cadence import generic, memories

RELEASED_OUTCOME = memories.EpisodicRecords.outcome
RELEASED_LEARN = generic.Brain.learn
RELEASED_COMPOSE = generic.Brain.compose

STATE: dict[str, Any] = {"rule": "off", "theta": 0.5, "writes": 0, "zeros": 0, "ctx_match": 0.9}


def _next_key(self: generic.Brain, next_observations: Any) -> np.ndarray | None:
    try:
        x = self._observations(next_observations)
        return np.asarray(self._code(x), dtype=float)
    except Exception:
        return None


def learn(self, reward, done, next_observations, *args, **kwargs):  # type: ignore[no-untyped-def]
    episodic = getattr(self, "_episodic", None)
    if episodic is not None and STATE["rule"] != "off":
        episodic._next_key = _next_key(self, next_observations)
    return RELEASED_LEARN(self, reward, done, next_observations, *args, **kwargs)


def _direct(records: memories.EpisodicRecords) -> dict[tuple[bytes, int], bool]:
    table = getattr(records, "_direct_table", None)
    if table is None:
        table = {}
        records._direct_table = table
    return table


def _slot_actions(action: np.ndarray, row: int) -> list[int]:
    a = np.asarray(action[row])
    return [int(v) for v in np.atleast_1d(a)]


def outcome(self, observations, action, reward, done, learned, *, gamma, woken=False, salience=None):  # type: ignore[no-untyped-def]
    rule, theta = STATE["rule"], float(STATE["theta"])
    observations = np.asarray(observations, dtype=float)
    reward = np.asarray(reward, dtype=float)
    done = np.asarray(done, bool)
    learned = np.asarray(learned, bool)
    marked = np.zeros(len(reward), bool) if salience is None else np.asarray(salience) > 0
    known = learned & ((reward != 0) | done | woken | marked)
    zero = learned & (reward == 0) & ~known
    extinguish = np.zeros(len(reward), bool)
    history = np.zeros(len(reward), bool)
    if rule != "off" and zero.any():
        STATE["zeros"] += int(zero.sum())
        if rule == "owner":
            held = self.held(observations, self.decided(len(action)), action)
            nxt = getattr(self, "_next_key", None)
            same = np.zeros(len(reward), bool)
            if nxt is not None and nxt.shape == observations.shape:
                same = np.all(np.isclose(nxt, observations), axis=1)
            extinguish = zero & (held >= theta) & same
        elif rule == "direct":
            table = _direct(self)
            held = self.held(observations, self.decided(len(action)), action)
            for row in np.flatnonzero(zero & (held >= theta)):
                k = observations[row].tobytes()
                extinguish[row] = all(table.get((k, a), False) for a in _slot_actions(action, row))
        elif rule == "history":
            table = _direct(self)
            keys = self.decided(len(action))
            if self.temporal is not None and keys is not None:
                held = self.held(observations, keys, action, memory=self.temporal)
                for row in np.flatnonzero(zero & (held >= theta)):
                    k = observations[row].tobytes()
                    history[row] = all(table.get((k, a), False) for a in _slot_actions(action, row))
        else:
            raise ValueError(f"unknown rule {rule!r}")
    if history.any():
        # the zero is written into the context records alone, under the decision's history
        STATE["writes"] += int(history.sum())
        keys = self.decided(len(action))
        memories.record_action(self.temporal, keys, action, np.zeros(len(reward)),
                               np.zeros(len(reward)), history, self.offsets)
    if extinguish.any():
        STATE["writes"] += int(extinguish.sum())
        tiny = np.where(extinguish, 1e-9, 0.0)
        salience = tiny if salience is None else np.where(extinguish, np.maximum(salience, 1e-9), salience)
    RELEASED_OUTCOME(self, observations, action, reward, done, learned, gamma=gamma, woken=woken,
                     salience=salience)
    if rule in ("direct", "history"):
        table = _direct(self)
        for row in np.flatnonzero(known & (reward > 0)):
            k = observations[row].tobytes()
            for a in _slot_actions(action, row):
                table[(k, a)] = True


def compose():  # type: ignore[no-untyped-def]
    """`Brain.compose` for the present rule: with `lib`, the library gene of
    P:/projects/cadence-210fix (`memory_extinction=theta`) on every memory_choice brain."""
    if STATE["rule"] not in ("lib", "libctx"):
        return RELEASED_COMPOSE
    theta = STATE["theta"]

    def composed(*args, **kw):  # type: ignore[no-untyped-def]
        if kw.get("memory_choice"):
            kw.setdefault("memory_extinction", theta)
        return RELEASED_COMPOSE(*args, **kw)

    return composed


RELEASED_EXTINCT = getattr(memories.EpisodicRecords, "extinct", None)


def _counted_extinct(self, observations, action, reward, known, learned, expected):  # type: ignore[no-untyped-def]
    out = RELEASED_EXTINCT(self, observations, action, reward, known, learned, expected)
    STATE["zeros"] += int((learned & ~known & (reward == 0)).sum())
    STATE["writes"] += int(out.sum())
    return out


def _ctx_flag(self, action: np.ndarray) -> np.ndarray:
    """Per row: was this action paid directly after a history like this decision's?"""
    keys = self.decided(len(action))
    paid = getattr(self, "_paid_contexts", {})
    out = np.ones(len(action), bool)
    if self.temporal is None or keys is None:
        return out
    keys = np.asarray(keys, float)
    for row in range(len(action)):
        k = keys[row]
        norm = np.linalg.norm(k)
        best = 0.0
        for a in _slot_actions(action, row):
            stored = paid.get(a)
            if not stored or norm == 0:
                best = 0.0
                break
            m = np.asarray(stored)
            sims = m @ k / (np.linalg.norm(m, axis=1) * norm + 1e-12)
            best = float(sims.max())
        out[row] = best >= STATE["ctx_match"]
    return out


def _ctx_extinct(self, observations, action, reward, known, learned, expected):  # type: ignore[no-untyped-def]
    out = RELEASED_EXTINCT(self, observations, action, reward, known, learned, expected)
    if out.any():
        out = out & _ctx_flag(self, np.asarray(action))
    STATE["zeros"] += int((learned & ~known & (reward == 0)).sum())
    STATE["writes"] += int(out.sum())
    return out


def _ctx_outcome(self, observations, action, reward, done, learned, *, gamma, woken=False, salience=None):  # type: ignore[no-untyped-def]
    reward_a = np.asarray(reward, float)
    done_a = np.asarray(done, bool)
    learned_a = np.asarray(learned, bool)
    marked = np.zeros(len(reward_a), bool) if salience is None else np.asarray(salience) > 0
    known = learned_a & ((reward_a != 0) | done_a | woken | marked)
    keys = self.decided(len(action)) if self.temporal is not None else None
    RELEASED_OUTCOME(self, observations, action, reward, done, learned, gamma=gamma, woken=woken,
                     salience=salience)
    if keys is not None:
        paid = getattr(self, "_paid_contexts", None)
        if paid is None:
            paid = {}
            self._paid_contexts = paid
        for row in np.flatnonzero(known & (reward_a > 0)):
            for a in _slot_actions(np.asarray(action), row):
                paid.setdefault(a, []).append(np.asarray(keys[row], float).copy())


def apply(rule: str, theta: float = 0.5) -> None:
    STATE.update(rule=rule, theta=float(theta), writes=0, zeros=0)
    if RELEASED_EXTINCT is not None:
        memories.EpisodicRecords.extinct = {"lib": _counted_extinct, "libctx": _ctx_extinct}.get(
            rule, RELEASED_EXTINCT)
    if rule in ("lib", "libctx") and RELEASED_EXTINCT is None:
        raise RuntimeError("rule lib needs the cadence-210fix library on PYTHONPATH")
    if rule == "libctx":
        memories.EpisodicRecords.outcome = _ctx_outcome
        generic.Brain.learn = RELEASED_LEARN
    elif rule in ("off", "lib"):
        memories.EpisodicRecords.outcome = RELEASED_OUTCOME
        generic.Brain.learn = RELEASED_LEARN
    else:
        memories.EpisodicRecords.outcome = outcome
        generic.Brain.learn = learn
