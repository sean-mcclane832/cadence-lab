# Settle probe: does the brain's state carry the ball's direction? (One-Frame Breakout)

Script: [experiments/settle_probe.py](../experiments/settle_probe.py) (protocol and predictions
committed in 5621509 before any run; one amendment, b34b1bf, after a 1-block smoke test and before
the main run). Results: [runs/settle_probe/](../runs/settle_probe/) (`summary.json`, one JSON per
arm and seed). cadence-net 0.89.0, seeds 10-13, 150 blocks.

## Question

In [long training](long_training_breakout.md) every Cadence arm, recall off included, reaches ~13
bricks on One-Frame Breakout while a one-frame MLP stops near 6. The guess was that the settled
state carries the previous moment, so the ball's direction.

## What the brain actually has

Reading the 0.89.0 source first changed the question a bit. `Brain.compose(49, 3, modules=(32,))`
is sensory 0-48, association 49-80, prefrontal 81-112, motor 113-115, plus a `working_memory`
Trace (decay 0, amplitude 3) that copies last moment's association activation into the prefrontal
region as this moment's drive. That is an Elman-style context layer, and it is on in the
"norecall" arm. Each settle also starts warm from the last settled state. So there are two
candidate carriers: the explicit prefrontal copy and the warm start.

## Setup

- Players trained exactly as `classic/run.py` (its own `make`, `test`, constants). The block test
  scores reproduce the long run bit for bit on all 8 lives, so probing did not disturb training.
- Checkpoints before training and after 10, 30, 75, 150 blocks. A frozen copy plays fresh probe
  games (>= 3,000 moments). At each moment: true direction the ball just moved (dir4, 4 classes)
  and MinAtar's next move (`ball_dir`), the input frame, the working-memory trace, and the settled
  activations.
- Linear probe: multinomial logistic regression, L2, 5-fold CV grouped by episode.
- Controls on the same moments: `frame` (one frame), `two_frame` (frame + previous frame, the
  ceiling), and a shadow **untrained** brain of the same seed fed the exact same observation
  stream (amendment: the untrained player's own games are so short that the frame alone decodes
  direction at 0.95, so it had to be matched).
- Lesions at test time, weights untouched, 10 episodes: `wm0` zeroes the working-memory trace
  before every act, `cold` settles from rest each moment (warm start removed), `both`.

## Results (norecall, mean of 4 seeds; dir4 accuracy on held-out episodes)

| block | majority | frame | frame + trained state | frame + untrained state | two frames | state alone |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.54 | 0.95 | 0.98 | 0.98 | 0.99 | 0.98 |
| 30 | 0.31 | 0.82 | 0.97 | 0.97 | 0.99 | 0.96 |
| 150 | 0.28 | 0.71 | 0.98 | 0.97 | 0.99 | 0.96 |

Bricks per test episode under lesions (norecall, per seed 10/11/12/13):

| block | intact | wm0 | cold | both |
| --- | --- | --- | --- | --- |
| 30 | 9.6 / 7.9 / 3.1 / 7.3 | 0.4 / 1.5 / 3.0 / 1.6 | 9.6 / 6.4 / 3.1 / 9.6 | 0.4 / 1.5 / 3.5 / 2.5 |
| 150 | 9.3 / 8.0 / 6.5 / 16.8 | 2.0 / 1.7 / 3.7 / 1.3 | 8.4 / 7.0 / 6.5 / 14.7 | 2.0 / 0.8 / 3.7 / 0.7 |

pos32 (recall + position term), 150 blocks, means: frame 0.71, frame + trained state 0.95, frame
+ untrained state 0.94, state alone 0.83 (falling from 0.92 at 30). Bricks intact 15.3, wm0 2.1,
cold 15.4, both 2.4. MinAtar's next-move target (`ball_dir`) tells the same story everywhere
(norecall 150: frame 0.62, frame + state 0.96, two frames 0.99).

## Predictions (norecall, 150 blocks)

- **P1 holds.** State beats frame on dir4 by 22-29 points, 4/4 seeds.
- **P2 holds.** Frame + working-memory trace beats frame by 24-31 points, 4/4.
- **P3 holds only on paper.** The matched untrained brain also beats frame by 23-31 points, and
  trained beats untrained in 4/4 seeds, but by 0.1-1.2 points. The direction is in the state by
  construction, not by learning.
- **P4 fails.** State-probe accuracy at 150 >= 30 in 1/4 seeds; Spearman of accuracy vs bricks over
  trained checkpoints is -0.43 (p = 0.10; pos32 -0.80). Accuracy does not rise with training or
  track the score. Part of the negative sign is the game: longer games mean more bounces, so the
  frame alone gets less predictive (0.82 -> 0.71) and everything shifts.
- **P5 holds, more strongly than predicted.** Zeroing the working-memory copy drops bricks 43-92%
  in 4/4 seeds, to about 2, *below* the one-frame MLP (~6): the trained brain depends on the copy.
  Removing the warm start costs little (10.2 -> 9.2 mean; pos32 none).

## Reading

- **The hypothesis is right about where, and wrong about how.** The ball's direction is linearly
  readable from the settled state, near the two-frame ceiling, and the brain uses it: take away
  the prefrontal copy and play collapses. But the carrier is not "the settle" in a loose sense.
  It is the explicit working-memory Trace that `compose` wires in by default, and the warm start
  does almost nothing.
- **Training does not build the memory, it learns to read it.** An untrained brain fed the same
  frames carries the direction just as well. What 150 blocks buy is a policy that uses it, which
  is the reservoir-computing picture: a fixed random copy of the last moment plus a learned
  readout. "Recall off" never meant "no memory of a moment ago".
- **pos32 is odd.** Its settled state alone gets *less* direction-readable with training
  (0.92 -> 0.83) while it plays better (15 bricks) and still dies under wm0. Its decision seems to
  combine the frame with the copy in a way a linear probe on the state alone sees less of. With 4
  seeds I would not lean on it.
- **Caveats.** Linear probes only; 4 seeds; 10 lesion episodes per cell; probe games come from the
  trained player's own policy, so the trained and untrained probes share the moments but the
  moments shift with training. The lesions are test-time only: a brain trained without the copy
  might learn something else.

## Next

- Train a brain with the copy silenced from birth (`compose` has no plain off switch for the
  prefrontal region; `working_memory_amplitude=0` should feed it zeros) through classic/run.py.
  If it stops near 6, the copy is the whole story behind 13 vs 6.
- Fairer baseline: give the MLP an Elman context (its own last hidden layer) instead of the raw
  second frame.
