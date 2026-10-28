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

STATE: dict[str, Any] = {"rule": "off", "theta": 0.5, "writes": 0, "zeros": 0}


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
    if STATE["rule"] != "lib":
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


def apply(rule: str, theta: float = 0.5) -> None:
    STATE.update(rule=rule, theta=float(theta), writes=0, zeros=0)
    if RELEASED_EXTINCT is not None:
        memories.EpisodicRecords.extinct = _counted_extinct if rule == "lib" else RELEASED_EXTINCT
    if rule == "lib" and RELEASED_EXTINCT is None:
        raise RuntimeError("rule lib needs the cadence-210fix library on PYTHONPATH")
    if rule in ("off", "lib"):
        memories.EpisodicRecords.outcome = RELEASED_OUTCOME
        generic.Brain.learn = RELEASED_LEARN
    else:
        memories.EpisodicRecords.outcome = outcome
        generic.Brain.learn = learn
