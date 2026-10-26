"""A two-armed bandit whose paying arm moves once, mid-life."""

from __future__ import annotations

import numpy as np


class ReversalBandit:
    """One constant observation, two arms.

    Before `reversal` arm 0 pays; from `reversal` on arm 1 pays (and from `back` on,
    if given, arm 0 again). The paying arm
    returns +1 with probability `p` (else -1); the other arm returns -1 with
    probability `p` (else +1). With p=1.0 the world is deterministic.
    """

    def __init__(self, life: int = 600, reversal: int = 300, p: float = 1.0, seed: int = 0,
                 back: int | None = None):
        self.life = life
        self.reversal = reversal
        self.back = back  # optional second flip: arm 0 pays again from `back` on (A-B-A)
        self.p = p
        self.rng = np.random.default_rng(seed)
        self.observation = np.ones((1, 1))

    def paying_arm(self, moment: int) -> int:
        if self.back is not None and moment >= self.back:
            return 0
        return 0 if moment < self.reversal else 1

    def pull(self, moment: int, arm: int) -> tuple[float, bool]:
        """Return (reward, hit) for pulling `arm` at `moment`."""
        hit = arm == self.paying_arm(moment)
        lucky = self.rng.random() < self.p
        good = hit == lucky
        return (1.0 if good else -1.0), hit
