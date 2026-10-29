# Serial reversal (v1)

> **Correction (2026-10-29, [recheck](serial_reversal_recheck.md), cadence 0.91.1, 12 seeds):** the
> curve reproduces exactly, and it is not the built-in working-memory copy (silenced from birth:
> slope still negative 12/12), not age, and not noise. But "learning-set curve" overstates it. About
> half the drop is savings: switching back to an arm that has paid before is faster without any
> reversal practice (a 2-flip life with matched arm histories: 5.25 errors vs 9.00 naive). The extra flips add
> about 3 errors on top (2.17 vs 5.25, 12/12 seeds), and part of that sits in the arousal state's
> running references, not in a learned switch rule. Read "Reading" below with that in mind.

Does a Cadence brain get faster at reversals with practice? Animals do (Harlow's learning sets;
macaques end near one error per reversal). Deep nets trained continually tend not to (loss of
plasticity, Dohare et al. 2024). The single-reversal bandit (`notes/bandit_reversal.md`) found the
continuing brain slower than a newborn and no savings on A-B-A; this asks what happens over many flips.

World (`envs/serial.py`): two arms, one constant observation, the paying arm flips 10 (or 30) times.
First block 200 moments, later blocks 100 +/- 20 (jittered so flips can't be timed; same schedule for
every arm of a seed). Reward +1 for the paying arm, -1 otherwise (or 0 with `--fail 0`). 12 seeds,
Cadence 0.89.0, CPU float64. Code: `experiments/serial_reversal.py`. All receipts in
`runs/serial_reversal/` verify with `cadence.Receipt.verify`.

Arms: `continuing` (`Brain.compose(1, 2, modules=(16,), arousal=True)`), `memory_choice` (+ `memory_choice=True`),
`memory_off` (`episodic=False`), `newborn` (fresh brain at every flip), `cue` (observation also carries
the last arm and whether it paid: `[1, arm0, arm1, won]`), `cue_choice` (cue + memory_choice), `mlp`
(1-16-2 Q-learner, as before), `mlp_cue`, `mlp_cbp` (MLP + continual-backprop-style reset of the least
useful hidden unit every 25 moments), `wsls` (hand-coded win-stay/lose-shift, the optimum: 1 error per
flip), `random`.

Metric: **errors** = wrong-arm choices in the first 20 moments after a flip. Slope = linear fit of
errors over reversal index, per seed.

## 1. Main: 10 flips, unpaid arm costs 1

| arm | rev 1 | rev 10 | first 3 | last 3 | slope | slope < 0 |
| --- | --- | --- | --- | --- | --- | --- |
| continuing | 8.1 | 3.3 | 7.19 | 3.53 | -0.56 | 12/12 |
| memory_choice | 9.1 | 3.4 | 8.36 | 3.64 | -0.68 | 12/12 |
| cue_choice | 11.0 | 2.3 | 9.42 | 2.94 | -0.97 | 12/12 |
| cue | 9.0 | 5.2 | 8.11 | 4.92 | -0.45 | 11/12 |
| memory_off | 18.2 | 13.1 | 16.08 | 14.50 | -0.22 | 9/12 |
| newborn | 2.2 | 1.2 | 1.97 | 1.58 | -0.05 | 7/12 |
| mlp | 8.6 | 8.2 | 8.72 | 9.31 | +0.09 | 6/12 |
| mlp_cue | 5.5 | 9.8 | 5.47 | 9.28 | +0.57 | 1/12 |
| mlp_cbp | 9.8 | 3.8 | 7.83 | 4.64 | -0.49 | 10/12 |
| wsls | 1.0 | 1.0 | 1.00 | 1.00 | 0 | - |
| random | 11.0 | 10.8 | 10.22 | 10.64 | +0.07 | 3/12 |

Every Cadence arm with the associative store gets faster with practice, in every seed. The MLP does
not; given the cue it gets *worse* (5.5 -> 9.8). Continual-backprop resets give the MLP a downward
curve too, matching Dohare et al.

**Equal blocks control** (`--first 100`, so the first block is no longer longer): same picture.
continuing 9.0 -> 3.8, slope -0.58, 12/12. memory_choice 9.7 -> 3.6, 12/12. cue_choice 11.5 -> 3.1, 12/12.
The drop isn't just a long first block.

## 2. Probe: does the speed survive a long stable stretch?

`--probe 400`: after the 10 flips plus one more ordinary block, one block of 400 moments, then a
final flip (reversal 12). Compared with a naive brain whose first and only block is 400 moments
(`--reversals 1 --first 400`).

| arm | naive, 400-moment block | experienced, after 400-moment block |
| --- | --- | --- |
| continuing | 7.6 | **2.4** |
| memory_choice | 9.0 | **2.8** |
| cue_choice | 10.8 | **2.7** |
| mlp | 6.1 | 10.7 |
| mlp_cbp | 5.9 | 4.8 |
| newborn | 2.2 | 1.8 |

The experienced Cadence brains stay fast after 400 stable moments, a third of the naive errors. The
learned flexibility is not just "preferences never got strong". The MLP is worse than naive.

## 3. Long life: 30 flips

| arm | rev 1-3 | rev 8-12 | rev 26-30 |
| --- | --- | --- | --- |
| continuing | 7.2 | 3.0 | 4.8 |
| memory_choice | 8.4 | 3.3 | 2.7 |
| cue_choice | 9.4 | 3.0 | **1.4** |
| mlp | 8.7 | 9.8 | 10.4 |
| mlp_cbp | 7.8 | 4.5 | 4.0 |
| newborn | 2.0 | 1.6 | 1.8 |

- `cue_choice` keeps improving to about 1.4 errors per flip. The optimum is 1.0 (one burned
  choice is needed to notice a flip). It approaches win-stay/lose-shift.
- Plain `continuing` bottoms out around reversal 11-17 (about 2-3) and drifts back up to about 5 (4.8 over the last five):
  a U-shape, consistent with slow plasticity loss on top of the learning-set effect.
  `memory_choice` does not drift.
- The MLP gets slowly worse.

## 4. Noisy world (p = 0.9)

Mostly gone. continuing first3 11.4 / last3 11.1 (8/12 negative), memory_choice 11.9 / 9.4 (10/12),
cue_choice 12.0 / 10.3 (10/12). MLP clearly worse with practice (5.9 -> 11.0, 1/12). With 10% flipped
outcomes a single loss is weak evidence, so even the WSLS reference makes 2.6 errors.

## 5. Unpaid arm pays 0 (upstream issue #210)

Issue #210: with `memory_choice=True`, a paid record isn't extinguished when failures pay 0, so a
reversal is often not acquired. Over many flips (`--fail 0`, 10 and 30 reversals):

| arm | rev 1 | rev 10 | rev 26-30 | slope < 0 (30 flips) |
| --- | --- | --- | --- | --- |
| memory_choice | 18.1 | 14.7 | 14.4 | 5/12 |
| continuing | 11.0 | 7.2 | 9.1 (alternating about 7 and 11-13) | 5/12 |
| cue | 12.0 | 9.0 | 9.9 | 7/12 |
| **cue_choice** | 17.2 | 2.2 | **2.0** | 12/12 |
| mlp_cbp | 10.1 | 8.0 | 7.1 | 8/12 |

- The #210 failure persists over a whole life: `memory_choice` never learns its way out (about 15
  errors per flip at flip 30).
- Putting the last outcome into the observation fixes it: `cue_choice` falls from 17 to about 2,
  in 12/12 seeds. A plausible reading: the episodic records are keyed by place (observation), so
  "after a win" and "after a loss" become different places, and a zero outcome no longer has to
  overwrite the old arm's paid record. Not verified against the source.
- Plain `continuing` alternates: flips back to the first arm cost more than flips away from it.

## Reading

- Cadence brains with the associative store get faster at reversals in this task: errors per
  reversal halve over 10 flips, in every seed, with or without a cue, and the gain survives a long
  stable block. (Corrected 2026-10-29: I first called this the animal learning-set signature. The
  recheck finds about half of it is savings on previously paid arms plus age, and part of the
  rest is arousal tuning; see [the recheck](serial_reversal_recheck.md).)
- A plain online MLP shows the opposite (flat or worse, worse than naive after the probe), as the
  loss-of-plasticity literature predicts. Continual backprop partly fixes it.
- A newborn at every flip is still the fastest in absolute terms (about 2 errors). Its youth
  arousal makes it a strong explorer. The continuing brains close most of the gap and
  `cue_choice` reaches the newborn's level by about flip 15, without throwing its life away.
- The cue + episodic memory combination is the most interesting arm: closest to optimal at -1, and
  the only arm that escapes the #210 zero-payoff trap.

## Limits

One tiny task (two arms, one observation), 12 seeds, founder genes, no tuning, `modules=(16,)`. The
cue is hand-made by the experimenter, a brain doesn't build it itself. The MLP baselines are untuned.
Errors are counted over a fixed 20-moment window.
