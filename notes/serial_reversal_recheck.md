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

## Results (cadence 0.91.1, seeds 0-11, test errors in the first 20 moments after the test flip)

All three parts were rerun on the final script so the receipts verify (`controls.json`,
`lesions.json`, `wm0.json`). The reruns matched the first runs to the digit.

### Controls

| condition | continuing | memory_choice | serial lower, paired seeds (cont / mc) |
| --- | --- | --- | --- |
| naive | 9.00 | 9.67 | 12/12, 12/12 |
| stable (1,100 moments, no flip) | 7.42 | 8.92 | 12/12, 12/12 |
| **serial (flip 11)** | **2.17** | **3.25** | |
| both_pay | 4.42 | 3.08 | |
| both_unfav (flip to the arm it never used) | 7.83 | 8.25 | 11/12, 12/12 |
| both_fav (flip back to its favourite) | 2.42 | 1.50 | 4/12 (1 tie), 0/12 (3 ties) |
| coin (paying arm random per moment) | 16.00 | 15.67 | |
| tf200 (2 flips) | 6.50 | 7.00 | |
| tf500 (2 flips) | 5.25 | 6.58 | |
| **tf_matched (2 flips, arm histories as serial)** | **5.25** | **6.67** | 12/12, 11/12 (1 tie) |

Pre-flip errors (last 20 moments before every flip) are 0 in every serial life, both arms, all
seeds: no stability trade. In `both_fav` they are 1.4 / 2.2: after 100 moments of the other arm
the brain was still partly choosing its favourite, so that flip "back" was half done before it
started. That makes `both_fav` a weak comparison, and `tf_matched` (pre-flip 0) the clean one.

### Lesions at the test flip (mean over 12 seeds)

| brain | intact | store_clear | consolidated_zero | wm0 | arousal_reset | arousal_swap | store_swap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| continuing serial | 2.17 | 2.17 | 2.17 | 2.08 | 20.00 | 5.00 | 4.00 |
| continuing naive | 9.00 | 8.67 | 8.92 | 9.08 | 20.00 | 5.17 | 8.75 |
| memory_choice serial | 3.25 | 2.67 | 2.83 | 3.25 | 20.00 | 5.08 | 3.75 |
| memory_choice naive | 9.67 | 9.00 | 9.33 | 7.50 | 20.00 | 9.08 | 9.17 |

(continuing serial store_swap 4.00 is one seed at 19; the other 11 average 2.6.)

### Working-memory copy off from birth (v1 serial run, `working_memory_amplitude=0`)

continuing first3 7.28, last3 3.00, slope -0.599, negative 12/12. memory_choice 7.33 / 3.28,
-0.608, 12/12. v1 with the copy: 7.19 / 3.53 / -0.559 and 8.36 / 3.64 / -0.678.

### Predictions, scored

- P1 holds (2.17 is a bit faster than the 2.5-4.5 range; it is flip 11, not 10).
- P2 holds (9.00 / 9.67).
- P3 mostly holds: `stable` is not fast (7.42 / 8.92), but continuing gains 1.6 errors from age alone,
  just under the >= 8 I wrote.
- P4 as written says "non-reversal": both_pay 4.42 / 3.08 is <= 5. But the split shows why: it
  is all flips back to the favourite. Flips to the arm it never used stay near naive (7.83 / 8.25).
  `coin` is far worse than naive (16), against my prior that it would be fast.
- P5 holds clearly. The built-in copy is not the carrier: wm0 at test 2.08 vs 2.17, and with
  the copy silenced from birth the curve is unchanged, 12/12 in both arms.
- P6: `arousal_reset` breaks switching (20, every seed, both brains). `arousal_swap` moves part
  of the gap: continuing serial +2.83 of a 6.83 gap (41%, under my 50% line), and the naive brain
  given the serial arousal drops 3.83 (56%). memory_choice: +1.83 (29%) and -0.58.
- P7 holds: the store is not where the gap lives.
- P8 holds: pre-flip errors 0.
- A1/A2: `tf_matched` is 3.08 (continuing) and 3.42 (memory_choice) above serial, just past the
  A2 line, 12/12 and 11/12 seeds. tf200 -> tf500 falls only a little (6.50 -> 5.25, 7.00 -> 6.58).

## Verdict: partly

Rule check: `serial` does not beat every non-reversal control by 3 (both_fav is as fast or
faster), so by the pre-registered rule this is **partly**, not survives. With the both_fav caveat
above, my reading of what is left:

- **Not an artifact of the built-in copy.** Silencing it from birth leaves the curve untouched.
  The 10/28 problem does not repeat here. In a constant-observation task the copy has nothing to carry.
- **Not age, not noise.** An age-matched brain without flips is near naive, and the serial brain
  makes no extra errors inside blocks.
- **About half of the drop is not "learning that the rule flips".** Splitting the naive-to-serial
  gap (continuing 6.8 errors / memory_choice 6.4): age about 1.6 / 0.75, savings from a paid
  history on the destination arm (tf_matched) about 2.2 / 2.25, the extra flips about 3.1 / 3.4.
  Once the arm being switched back to has paid before, switching back is faster, with no
  reversal practice needed.
- **What the extra flips add is real, and part of it is arousal.** With arm histories matched,
  11 flips still beat 2 by about 3 errors in 23/24 seed pairs. Part of that sits in the arousal
  state (its running reward and error references): transplanting a naive brain's arousal into
  the serial brain costs it 41% (continuing) / 29% (memory_choice) of the gap, and the other way round
  a naive brain gains 56% (continuing). The arousal references are adapted by experience, so
  this is learning in a weak sense, a surprise gauge tuned by a history of rare losses. It is not a
  win-stay/lose-shift rule held in the weights. `coin` fits that: a life full of random losses
  turns the gauge down and the brain stops switching (16 errors, worse than naive).

So the v1 sentence "Cadence brains show a learning-set curve" overstates it. Better: Cadence
brains get faster at reversals with practice. About half of that is savings on arms that have paid
before (plus a little age). The rest comes from the flips themselves, partly through arousal
references tuned by rare, flip-locked losses. That is neither a pure artifact nor the animal
learning-set reading.

## Limits

- `tf_matched` matches how much each arm paid, not when: serial's arm-1 history is spread
  over recent blocks, tf_matched's is one early block. Some of the "extra flips" share may be
  recency of the destination arm's history, which would push more of the curve toward savings.
- One task, constant observation, 12 seeds, `modules=(16,)`, founder genes, one test flip.
- The swap lesions transplant the whole arousal state (age included); I did not split which
  reference matters. The cue arms were not retested; the cue is experimenter-made either way.
