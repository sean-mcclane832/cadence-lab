"""A two-armed bandit whose paying arm flips many times in one life (serial reversal)."""

from __future__ import annotations

import numpy as np


class SerialBandit:
    """One constant observation, two arms, the paying arm flips at each of `flips`.

    The first block is `first` moments long, the rest are jittered around `block`
    (block +/- jitter) so a learner can't time the flips. Block lengths come from
    `schedule_seed`, so every arm of one seed sees the same schedule. Rewards follow
    ReversalBandit: +1 for the paying arm with probability `p`, else `fail` (-1 by default;
    0 is the "unpaid, no punishment" case of upstream issue #210). With `probe`, one extra
    block of `probe` moments is added before one extra flip, to test whether fast switching
    survives a long stable stretch.
    """

    def __init__(self, reversals: int = 10, first: int = 200, block: int = 100, jitter: int = 20,
                 p: float = 1.0, seed: int = 0, fail: float = -1.0, probe: int | None = None):
        sched = np.random.default_rng(10_000 + seed)
        lengths = [first] + [int(block + sched.integers(-jitter, jitter + 1)) for _ in range(reversals)]
        if probe:
            lengths += [probe, block]
        self.flips = list(np.cumsum(lengths)[:-1].astype(int))
        self.life = int(sum(lengths))
        self.p = p
        self.fail = fail
        self.rng = np.random.default_rng(seed)
        self.observation = np.ones((1, 1))

    def paying_arm(self, moment: int) -> int:
        return int(np.searchsorted(self.flips, moment, side="right") % 2)

    def pull(self, moment: int, arm: int) -> tuple[float, bool]:
        hit = arm == self.paying_arm(moment)
        lucky = self.rng.random() < self.p
        good = hit == lucky
        return (1.0 if good else self.fail), hit
