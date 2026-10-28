# A classic animal-learning battery

Five famous effects from animal conditioning, each run as an operant analog on one continuing
Cadence brain (Cadence 0.89.0 as released, founder genes) and on a plain MLP Q-learner. The
question for each: does the brain show the effect animals show?

Upstream (muellerberndt/cadence docs, tests, benchmarks, cadence-demos) was searched first: none
of these effects is tested there. Code: `experiments/animal_battery.py`, which declares every
test (prediction, metric, seeds, controls) in its docstring, including the revisions made after
a pilot and after the first run. Receipt: `runs/animal_battery/receipt.json` (schema
`cadence-lab/animal-battery/v1`, verifies from a fresh clone). Every life's raw numbers are inside the receipt.

## The task

Each trial is two moments of one life through `live`: a stimulus moment, then an inter-trial
moment with no stimulus. Two actions, press or withhold. Observation: `[bias, A, B, C, context X,
context Y]`. A press on a reinforced trial pays +1. Any other press costs 0.5 (effort): an
unreinforced press, a press between trials, every extinction trial. Withholding pays 0.
"Press rate" is the share of stimulus trials on which the brain pressed, i.e. its own behaviour
(greedy when calm, sampled when aroused), read over the first 10 trials of a test session run in
extinction, as in an animal test.

Arms: `cadence` = `Brain.compose(6, 2, modules=(16,), arousal=True)`; `cadence_mc` = the same with
`memory_choice=True`; `mlp` = a 6-16-2 tanh Q-learner, epsilon-greedy 0.1, one SGD step per moment.
24 seeds per cell, paired by seed. An effect "shows" when the mean paired difference is in the
predicted direction by more than two standard errors.

## Results

| effect | what animals do | cadence | cadence + memory_choice | MLP |
| --- | --- | --- | --- | --- |
| **Kamin blocking** | pretrained A blocks learning about B | **yes**: B 0.31 vs 0.74 control, 20/24 | yes, weaker: 0.55 vs 0.71, 11/24 (9 ties) | yes, near floor: 0.06 vs 0.18, 24/24 |
| **Unblocking** | a bigger reward in phase 2 lets B be learned | **yes**: B 0.73 vs 0.31 blocked, 21/24 | yes: 0.71 vs 0.55, 16/24 | yes, near floor: 0.13 vs 0.06 |
| Partial reinforcement extinction (PREE) | 50% training extinguishes later | **no, the reverse trend**: 100 vs 121 trials (n.s.) | **reverse**: 150 vs 285 trials, 5/24 | no: 10.7 vs 10.8 |
| ABA renewal | response returns in the training context | no: 0.36 vs 0.35 (AAA) | no: 0.45 vs 0.51 | no: 0.12 vs 0.10 |
| Spontaneous recovery | response returns after time passes | **no, by construction** (24/24 identical) | no, identical | no (no time) |
| Latent inhibition (neutral pre-exposure) | pre-exposed stimulus is learned more slowly | no: 0.19 vs 0.17 | weak: 0.40 vs 0.54 early presses, shows; trials to acquire 65 vs 52, n.s. | no |

Details per effect follow. Numbers are means over 24 seeds.

### Kamin blocking: shown

Block group: A reinforced (200 trials), then the compound AB reinforced (200). Control: C
reinforced, then AB. Test: B alone, in extinction.

| arm | B, block | B, control | difference | seeds in direction | A alone (block group) |
| --- | --- | --- | --- | --- | --- |
| cadence | 0.31 | 0.74 | +0.43 +/- 0.06 | 20/24 (2 ties) | 0.76 |
| cadence_mc | 0.55 | 0.71 | +0.16 +/- 0.07 | 11/24 (9 ties) | 0.73 |
| mlp | 0.06 | 0.18 | +0.11 +/- 0.01 | 24/24 | 0.23 |

The default Cadence brain blocks clearly. Having already learned that A predicts the reward, it
learns little about B. The MLP blocks too, which is the textbook prediction-error result, but it
presses so little in any extinction test (it extinguishes in about 10 trials) that its numbers sit
near the floor.

### Unblocking: shown (the prediction-error control)

Added as v3 after blocking turned out to be the clean positive. Same as the block group, but the
compound AB pays +2 instead of +1 in phase 2, so the compound brings a surprise that B can
explain. A prediction-error learner should then learn B; a learner tracking co-occurrence
would treat both groups alike.

| arm | B, unblock (+2) | B, block (+1) | difference | seeds in direction |
| --- | --- | --- | --- | --- |
| cadence | 0.73 | 0.31 | +0.42 +/- 0.06 | 21/24 (2 ties) |
| cadence_mc | 0.71 | 0.55 | +0.16 +/- 0.06 | 16/24 (6 ties) |
| mlp | 0.13 | 0.06 | +0.07 +/- 0.01 | 16/24 (8 ties) |
| cadence, modules=(32,) | 0.70 | 0.37 | +0.33 +/- 0.07 | 18/24 (4 ties) |

With the surprise restored, B comes back to the unblocked control's level (0.73 against 0.74
in the blocking control). Together, blocking and unblocking pin the default Cadence brain's
learning down as prediction-error driven, the Rescorla-Wagner signature.

### Robustness: modules=(32,)

The default brain with a twice-larger module (arm `cadence32`, v3) gives the same picture on
every effect: blocking 0.37 vs 0.70 (20/24) and unblocking (above); no PREE (pree2: 110 vs 144
trials, reverse direction); no renewal (ABA 0.39, AAA 0.38); no spontaneous recovery (0.36 later
vs 0.38 now; here waiting changed 5 lives, in the wrong direction); latent inhibition v2
borderline on trials to acquire (186 vs 130, +56 +/- 28, just over two standard errors) and not
on early presses.

### Partial reinforcement extinction: not shown, and Cadence leans the other way

v1 is void: the 50% group never learned to press well (0.32 at the end of training), so its
extinction started low. v2 (declared after reading v1, before running it): press cost 0.25 for
this effect, the 50% group trains to a criterion (trailing 20 trials >= 0.8) plus 100 trials, and
the 100% group gets the same number of trials on the same seed. Then 600 extinction trials.
Both Cadence arms reached the criterion on 24/24 seeds and ended training at about 0.88 vs 1.0.

| arm | trials to extinction, 50% | 100% | difference | press rate, first 50, 50% | 100% |
| --- | --- | --- | --- | --- | --- |
| cadence | 100 | 121 | -21 +/- 23 | 0.62 | 0.64 |
| cadence_mc | 150 | 285 | **-136 +/- 42** | 0.62 | 0.74 |
| mlp | 10.7 | 10.8 | -0.1 +/- 0.3 | 0.10 | 0.15 |

No arm shows the animal effect. With `memory_choice=True` the continuously rewarded brain
extinguishes much more slowly, which is what a value learner predicts: under 100% reward the
learned value of pressing is higher, so it takes longer to fall below the cost. The default
brain is in between and noisy. The MLP extinguishes so fast (about 10 trials) that the test
can't separate anything. (The MLP met the criterion but drifted back to 0.48 by the end of
training under its 10% random choices, so its row is weak either way.)

### ABA renewal: not shown

Acquire A in context X (200 trials), extinguish A to a criterion in context Y (trailing 10 <= 0.2;
every life reached it), test A in X. Controls AAA and ABB.

| arm | ABA | AAA | ABB | ABA - AAA | ABA - ABB |
| --- | --- | --- | --- | --- | --- |
| cadence | 0.36 | 0.35 | 0.33 | +0.01 +/- 0.04 | +0.03 +/- 0.02 |
| cadence_mc | 0.45 | 0.51 | 0.37 | -0.06 +/- 0.05 | +0.07 +/- 0.02 |
| mlp | 0.12 | 0.10 | 0.08 | +0.03 +/- 0.02 | +0.05 +/- 0.01 |

The key comparison is ABA vs AAA, and no arm has it. The small ABA > ABB difference in
`cadence_mc` and the MLP means only that testing in the extinction context is a little lower
than testing elsewhere, which isn't renewal.

### Spontaneous recovery: not shown, structurally

After extinction to a criterion, the "later" group waits 500 moments (`Brain.wait` with the
blank observation, while the last action awaits its outcome) before the test. All 24 seeds in
both Cadence arms behave identically to the "now" group. A direct check found no brain state
that differs after the waits: the working trace settles to the same point for a blank input
within a moment, and a wait changes no parameters, critic or associative memory by design. So
time alone can't bring a response back in 0.89.0. In animals it does.

### Latent inhibition: weak at best

v1 is void: a press during pre-exposure cost 0.5, so pre-exposure trained the brain not to
press to A. That is response learning, not latent inhibition, and it showed up as a strong
"effect" (cadence_mc 118 vs 52 trials to acquire). v2: pre-exposure has no consequence (an A
moment pays 0 whatever the action), then A is reinforced for 400 trials.

| arm | early presses (first 50), pre | none | trials to acquire, pre | none |
| --- | --- | --- | --- | --- |
| cadence | 0.19 | 0.17 | 223 | 189 (+34 +/- 30) |
| cadence_mc | 0.40 | 0.54 (shows, 14/24) | 65 | 52 (+13 +/- 9) |
| mlp | 0.55 | 0.48 | 34 | 41 |

Only `cadence_mc` leans the animal way, and only on the early press rate, not on the trials to
acquire. Weak. The default brain acquires slowly in both groups here (many lives take most of
the 400 trials).

## Reading

- **Blocking and unblocking are the clean positive.** One continuing Cadence brain shows Kamin
  blocking strongly (20 of 24 seeds), and raising the reward in phase 2 undoes it (21 of 24).
  That pair is the signature of learning from prediction error rather than from co-occurrence,
  and it holds at modules=(32,).
- **The extinction-side effects all fail.** PREE, renewal and spontaneous recovery are the
  effects that made animal-learning theory decide extinction is *new learning* (an inhibitory
  memory tied to context and time) rather than erasing the old association. Cadence behaves
  like an erasing value learner: the response doesn't come back with a context change or with
  time, and continuous reward makes extinction slower, not faster.
- That fits #210 and the docs: Cadence has no separate extinction memory, and its context
  drift (`wait`) doesn't touch what the choice reads. A brain that kept "A paid in X" and "A
  stopped paying in Y" as separate context-keyed records might show renewal. That's a concrete
  design question for the library, not something this battery can settle.
- The MLP is a weak baseline here (it extinguishes in about 10 trials in every design), so most
  of the "MLP vs Cadence" contrasts are floor effects. Blocking is the only effect where both
  learn enough to compare.

## Limits

Operant analogs of Pavlovian designs (the brain chooses to press; animals in most of these
classics are measured on conditioned responses). One task geometry, 24 seeds, founder genes, no
tuning, the press cost chosen in a pilot (0.5; 0.25 for PREE v2). Test sessions are short (10
trials) and run in extinction, so the test itself can extinguish. Two v1 designs were confounded
and redone (both versions are in the receipt). The MLP baseline is untuned.
