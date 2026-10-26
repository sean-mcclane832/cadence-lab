# Bandit reversal (v1)

Two-armed bandit, one constant observation, 600 moments. Arm 0 pays until moment 300, then
arm 1 pays. Reward +1 for the paying arm, -1 for the other. `p` is reward reliability (with
`p=0.9` each outcome is flipped with probability 0.1). 12 seeds (0-11), Cadence 0.89.0, CPU
backend (NumPy + Numba, float64). Receipts: `runs/bandit_reversal/receipt.json` (p=1.0) and
`receipt_p0.9.json`, both verified with `cadence.Receipt.verify`.

Arms:

| Arm | What |
| --- | --- |
| continuing | `Brain.compose(1, 2, modules=(16,), arousal=True)` lives through the reversal |
| cold_reset | same brain, `brain.reset(keep=())` at the reversal (arousal references cleared, parameters and memory kept) |
| newborn | a fresh brain (new seed) replaces the old one at the reversal |
| memory_off | continuing, but `episodic=False` (no associative store) |
| mlp | PyTorch MLP 1-16-2, Q-values, epsilon-greedy 0.1, one SGD step (lr 0.05) per moment |
| random | uniform random arm |

Metrics: hit rate (paying arm chosen) over the 50 moments before the reversal (`pre`), the 50
after it (`post50`) and the last 50 of life (`last`); mean reward over the 300 post-reversal
moments; moments to recover = first moment after the reversal where the trailing 20-moment hit
rate reaches 0.9 (so 20 is the floor). Work = answer sweeps + learning sweeps over the life.

## p = 1.0 (deterministic)

| Arm | pre | post50 | last | post reward | recovered | median recover (range) | sweeps | aroused share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| continuing | 1.00 | 0.79 | 1.00 | 0.93 | 12/12 | 35.5 (24-55) | 5,622 | 0.25 |
| cold_reset | 1.00 | 0.00 | 0.00 | -1.00 | 0/12 | never | 4,057 | 0.17 |
| newborn | 1.00 | 0.93 | 1.00 | 0.97 | 12/12 | 20 (20-29) | 7,264 | 0.33 |
| memory_off | 1.00 | 0.20 | 1.00 | 0.53 | 12/12 | 122 (78-182) | 7,680 | 0.36 |
| mlp | 0.96 | 0.65 | 0.94 | 0.80 | 12/12 | 25 (22-67) | n/a | n/a |
| random | 0.51 | 0.47 | 0.51 | 0.02 | 1/12 | (chance) | n/a | n/a |

## p = 0.9 (noisy)

| Arm | pre | post50 | last | post reward | recovered | median recover (range) | sweeps | aroused share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| continuing | 0.98 | 0.69 | 1.00 | 0.68 | 12/12 | 41 (28-150) | 6,732 | 0.29 |
| cold_reset | 0.98 | 0.00 | 0.00 | -0.81 | 0/12 | never | 4,470 | 0.18 |
| newborn | 0.98 | 0.84 | 1.00 | 0.73 | 12/12 | 20.5 (20-41) | 8,176 | 0.36 |
| memory_off | 1.00 | 0.24 | 0.94 | 0.26 | 11/12 | 163 (67-277) | 8,299 | 0.46 |
| mlp | 0.95 | 0.78 | 0.94 | 0.68 | 12/12 | 25.5 (22-49) | n/a | n/a |
| random | 0.51 | 0.47 | 0.51 | 0.00 | 1/12 | (chance) | n/a | n/a |

## Reading

- The continuing brain relearns the reversal in every life (12/12 at both noise levels) and ends
  at 1.00. The associative store does the work: switching it off (`memory_off`) makes recovery
  3-4x slower (122 vs 35.5 moments) and halves post-reversal income.
- `reset(keep=())` is the worst arm: the brain wakes calm, never feels the loss and keeps the old
  arm for the whole second half (0/12). This matches the Cadence docs (`docs/start.md`, "Reset and
  save"). Keeping the arousal state through a reset is what lets a brain notice a change.
- **The continuing brain does not win on speed.** A newborn brain (100 aroused "youth" moments)
  and the online MLP both recover faster (20 and 25 moments vs 35.5). In the noisy world the MLP
  ties the continuing brain on post-reversal income (0.68) and the newborn beats it (0.73).
  On this task, carrying the old life costs some recovery time; there is no old skill here worth
  keeping, so the test can't show the upside of continuity.
- Work: the continuing brain is calm 75% of its life and spends fewer sweeps than the newborn
  or memory-off arms.

## A-B-A (old rule comes back)

Same setup, p=1.0, 900 moments: arm 0 pays, arm 1 from moment 300, arm 0 again from 600
(`--life 900 --reversal 300 --back 600`, receipt `runs/bandit_reversal/receipt_aba.json`). The
first flip matches the table above exactly (same seeds). Second flip, back to arm 0:

| Arm | first 50 after return | recovered | median recover (range) | reward after return | life sweeps |
| --- | --- | --- | --- | --- | --- |
| continuing | 0.83 | 12/12 | 36.5 (23-50) | 0.94 | 7,280 |
| newborn (born at 300) | 0.84 | 12/12 | 27 (21-43) | 0.94 | 9,214 |
| memory_off | 0.26 | 12/12 | 97.5 (65-131) | 0.64 | 10,841 |
| mlp | 0.59 | 12/12 | 26 (20-104) | 0.74 | n/a |
| cold_reset | 1.00 | 12/12 | 20 | 1.00 | 4,057 |
| random | 0.51 | 0/12 | never | 0.01 | n/a |

- No savings from continuity: the continuing brain takes about as long to go back to A (36.5)
  as it took to leave it (35.5). Relearning an old rule is not faster here.
- It ties the newborn on income after the return (0.94) and beats the MLP (0.74) and memory-off
  (0.64), at lower work than the newborn.
- cold_reset "wins" only because it never left arm 0. Not a real result.

## Limits

One tiny task, 12 seeds, one composition (`modules=(16,)`), founder genes, no tuning. The MLP
is a simple baseline, not tuned either. Interesting next steps: contextual cues (several
observations), the `need`/`need_scale` genes, and `memory_choice=True`.
