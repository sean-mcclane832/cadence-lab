# Serial reversal recheck: learning to learn, or mechanics?

Question: the [serial reversal note](serial_reversal.md) (2026-10-27, cadence 0.89.0) read the drop in
errors per flip (continuing 8.1 -> 3.3 over 10 flips, 12/12 seeds) as a learning-set curve. On
10/28 the [settle probe](settle_probe.md) showed that a "learned" memory in One-Frame Breakout was
really the built-in working-memory copy. Does something built-in explain this curve too?

Script: [experiments/serial_reversal_recheck.py](../experiments/serial_reversal_recheck.py).
Results: `runs/serial_reversal_recheck/`. CPU, float64, one thread per process.

## Version

The 10/27 receipts were made on cadence 0.89.0. The upstream clone is now 0.91.1. First step, the
v1 command rerun on 0.91.1 (`repro_0911.json`, 12 seeds): continuing 8.1 -> 3.3 (first3 7.19, last3
3.53, slope -0.559, 12/12), memory_choice 9.1 -> 3.4 (12/12), cue_choice 11.0 -> 2.3 (12/12),
memory_off 16.08 / 14.50 (9/12). Identical to the 0.89.0 table to the printed digit, so every number
below is on 0.91.1 and comparable to v1.

## Mechanisms that could make later flips faster without learning to learn

Read from the 0.91.1 source (`generic.py`, `stream.py`, `memory.py`, `arousal.py`):

1. **Working-memory copy.** `compose` wires a Trace (decay 0, amplitude 3) that copies last moment's
   association activation into the prefrontal region. In this task the observation is a constant
   `[1]`, so the copy holds no outside information, only last moment's settled state (which the
   motor feedback ties to the last choice). It is there from birth and does not change with age,
   so by itself it cannot make flip 10 faster than flip 1. It could be a substrate a learned
   lose-shift rule uses.
2. **Arousal.** Youth arousal covers the first 100 moments; after that an outcome or surprise
   against running references (recent and long-run reward, usual errors, reward scale) rouses the
   brain and it samples. Those references adapt over a life. If a volatile life just tunes them so
   a drop in reward rouses the brain sooner, the speed-up is a built-in volatility gauge, not a
   learned rule.
3. **The associative store** (`hippocampus`, fast records plus consolidated records, consolidation
   0.05, memory_decay 0.9). The key is the constant observation, so it is one record per arm. After
   many flips both arms have paid records, so neither dominates and the brain can tip quickly.
   That is preference balancing, which any history of both arms paying would give, flips or not.
4. **Slow weights balancing / entrenchment.** The first block entrenches arm 0; alternating blocks
   wash that out. Same logic as 3, in the cortex weights. (v1's equal-blocks control removed only
   the longer first block, not this.)
5. **Age** (anything that changes with moments lived, flips or not: youth ending, weight growth or
   saturation, arousal age).
6. **Eligibility traces / TD credit.** Short lived (a few moments); cannot carry across blocks.
7. **Stability trade.** If the brain just gets noisier, post-flip errors fall but pre-flip
   errors rise. Checked by counting errors in the last 20 moments before each flip.
8. **The cue arms.** `cue` and `cue_choice` hand the brain its last arm and outcome in the
   observation. That input is experimenter-made, as v1 said. The working-memory copy then adds the
   moment before that. Not a hidden artifact, but the "near optimal" claim rests on a hand-made
   input. This recheck tests the uncued brains.

## Tests (pre-registered, written before the main runs)

Every control ends with the same test: a 100-moment block where arm 0 pays, then a flip to arm 1;
errors = wrong choices in the first 20 moments after the flip. Arms `continuing` and
`memory_choice`, seeds 0-11.

| condition | life before the test block | what it holds fixed |
| --- | --- | --- |
| naive | nothing | baseline |
| stable | 1,100 moments of arm 0 only (age matched, no flip) | mechanism 5 |
| serial | the v1 schedule, 10 flips (test = flip 11) | the claim |
| both_pay | 1,000 moments where both arms pay +1 | 3, 4: both arms valued, no losses, no flips |
| coin | 1,000 moments where the paying arm is drawn per moment | 3, 4, 2: both arms, losses, volatility, no block structure |

Lesions at the test flip (the brain copied, the 20 test moments replayed): `intact`, `store_clear`
(fast and consolidated records erased), `consolidated_zero`, `wm0` (the copy's amplitude set to 0
for the window), `arousal_reset`, `arousal_swap` (the serial brain gets the naive brain's arousal
state and vice versa), `store_swap` (same for the store). Plus a full v1 serial run with
`working_memory_amplitude=0` from birth.

Disclosure: one smoke run of the lesion part on seed 0, `continuing` only, was seen before writing
these predictions: serial intact 2, store_clear 2, consolidated_zero 2, wm0 1, arousal_reset 20;
naive 10, 10, 10, 10, 20.

### Predictions

- P1. `serial` reproduces v1's late flips: mean 2.5-4.5 test errors, both arms.
- P2. `naive` about 9 (v1 equal-first flip 1 was 9.0).
- P3. `stable` is no faster than `naive` (mean >= 8). If it is <= 5, age alone explains the curve.
- P4 (the decisive one). If the curve is learning from reversals, `both_pay` and `coin` stay near
  naive (>= 7). If either is <= 5, the speed-up comes from having lived with both arms paying or
  with volatile outcomes, not from reversals. My honest prior: `coin` will be fairly fast.
- P5. The working-memory copy is not the carrier: `wm0` at test changes serial errors by less than
  1.5 on average, and the wm0-from-birth serial run keeps a negative slope in >= 10/12 seeds.
- P6. `arousal_reset` breaks switching in both (about 20); it says arousal is the switch, not
  where the gap lives. `arousal_swap` locates the gap: if the serial brain with naive arousal rises
  by at least half the serial-naive gap, the speed-up lives in the arousal references.
- P7. Store: `store_clear` and `store_swap` move continuing by < 1.5 (seed 0 smoke); memory_choice
  more, direction not predicted.
- P8. No stability trade: pre-flip errors stay <= 1 per 20 moments in late blocks.

Verdict rule: **survives** if `serial` beats every non-reversal control (stable, both_pay, coin) by
>= 3 errors on average and in >= 9/12 seeds (paired), and the gap does not sit wholly in a
built-in, non-learned state. **Partly** if a non-reversal control (both_pay or coin) gets within
1.5 errors of serial, or one built-in state (arousal references) carries most of the gap: then
the brain gets faster with practice, but "learned that the rule flips" is the wrong reading.
**Retracted** if `stable` (pure age) matches serial, or the wm0 copy is the carrier.

## Results

(pending)

## Amendment 1 (written after the first controls, before these runs)

What the first controls showed: in `both_pay` the brain locks onto one arm (arm-1 choice share
0.02-0.05 or 0.81-0.98 per seed), so whether the test flip went *back* to that favourite decided
the result. Two extra conditions (`both_fav`, `both_unfav`) split it, and they suggest a simpler
mechanism than learning to learn: **savings**, a flip back to an arm with a long paid history is
fast, with no reversal experience needed.

At the serial test flip (flip 11, to arm 1), arm 1 has about 500 paid moments behind it and arm 0
about 600. New conditions, two flips per life, test flip back to arm 1:

| condition | arm 1 pays | then arm 0 pays | then flip to arm 1 |
| --- | --- | --- | --- |
| tf200 | 200 | 100 | test |
| tf500 | 500 | 100 | test |
| tf_matched | 500 | 600 | test (histories matched to serial, 2 flips instead of 11) |

Predictions:
- A1. Savings reading: `tf_matched` within 1.5 errors of `serial` (2.17 / 3.25), and errors fall
  from `tf200` to `tf500`.
- A2. Learning-to-learn reading: `tf_matched` at least 3 errors above `serial`.
- In between (1.5-3 above serial): both contribute.
