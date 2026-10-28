# Issue #210 checks: songs, key-door, and an extinction rule that keeps them

Upstream [#210](https://github.com/muellerberndt/cadence/issues/210): with `memory_choice=True`, a
paid episodic record is never extinguished when failures pay 0, so a reversal is often not acquired.
The owner's gene (`memory_extinction=theta`: write a learned zero when the vote held for the
action is >= theta and the observation did not change) fixed the bandit but broke songs, the
key-door and the odour nursery through `step`, and was dropped for a documentation-only fix.

This note: (1) the owner's preservation checks rerun on 0.89.0, (2) the cue workaround from
`notes/serial_reversal.md` through them, (3) a narrower rule, built as a library gene in a
separate copy (`P:/projects/cadence-210fix`, branch `issue210-direct-extinction`, not pushed
anywhere), checked on the same panel and on fresh seeds.

Code: `experiments/issue210/` (`rules.py` the rules as in-process patches, `chambers.py` bandit
and songs, `keydoor.py`, `odour.py`, `chain.py` wrappers that run the upstream chambers
unchanged except for `memory_choice=True`). Every run was declared in its file's docstring before
it ran. Cadence 0.89.0 (`81345d8`), CPU float64.

## The rules

| rule | writes a learned zero outcome when |
| --- | --- |
| `off` | never (0.89.0 as released) |
| `owner` | the vote for the chosen action >= 0.5 and the next observation equals this one (the owner's gene as described on #210, rebuilt here) |
| `direct` | the vote >= 0.5 and **a known outcome of this action itself has paid in this place** (reward > 0; not replay, not a demonstration) |
| `history` | as `direct`, but judged and written in the context records only (the place record is left) |
| `lib` | `direct` as a library gene, `Brain.compose(..., memory_choice=True, memory_extinction=0.5)`, the direct payments kept in a persistent associative store keyed like the place records; a place counts as paid when that store reads >= 0.9 of a whole payment (see below) |

The idea behind `direct`: in every chamber the owner's gene broke, the record it erased was built
by **replay** (key-door corridor, odour avoidance) or by a **demonstration** (songs). In the
#210 bandit, the old arm's record was built by **its own payments**. So: extinguish only a
payment that stopped, never credit the action was given by replay or a teacher.

## Results, development seeds

The issue's bandit (A 120, B 120, A 120 trials, one observation, `live`), 12 seeds:

| rule | fail | B acquired | B lag (median) | return acquired | zero writes / zero outcomes |
| --- | --- | --- | --- | --- | --- |
| off | 0 | **5/12** | 106 | 12/12 | 0 / 0 |
| off + cue | 0 | 4/12 | 84.5 | 12/12 | 0 / 0 |
| owner | 0 | 11/12 | 84 | 12/12 | 64 / 641 |
| direct | 0 | **11/12** | 84 | 12/12 | 64 / 641 |
| history | 0 | 10/12 | 97 | 12/12 | 66 / 805 |
| lib | 0 | **11/12** | 84 | 12/12 | 64 / 641 |
| all rules | -1 | 12/12 | 34.5 | 12/12 | 0 / 0 |

Songs (a **reconstruction**: the owner's songs chamber is not in the repository). Two songs of
four notes, A = 0 1 2 3, B = 3 2 1 0, cued once then three identical "continue" inputs, ten
lessons through `step(teacher=...)` with zero outcomes, greedy playback. Gate as the owner's:
control (`memory_choice=False`) + 0.10. 12 seeds:

| arm | mean notes right | gate 0.600 | zero writes |
| --- | --- | --- | --- |
| control, memory_choice=False | 0.500 | - | - |
| memory_choice, off | 0.625 | pass | 0 |
| memory_choice, owner | **0.573** | **fail** | 360 |
| memory_choice, direct / history / lib | 0.625 | pass | 0 |
| memory_choice + cue (last note in the observation) | 0.802 | pass | 0 |

The owner's songs failure reproduces in the reconstruction (pass -> fail), for the reason he gave:
the zero after each taught note on a "continue" input erases it.

Key-door (upstream `key_door.py`, protocol-3, chest -> lamp -> chest, `live`), seeds 0-7, trips to
acquire / adapt / return (median) and lives returning within 50 trips:

| delay | rule | acquire A | adapt B | return A | returns <= 50 | zero writes |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | off | 10.0 | 10.5 | 20.5 | 7 of 8 | 0 |
| 2 | owner | 9.5 | 9.5 | **49.5** | **4 of 8** | 871 |
| 2 | direct / history | 10.0 | 10.5 | 20.5 | 7 of 8 | 0 |
| 2 | off + cue | 26.5 | 18.0 | 58.5 | 4 of 8 | 0 |
| 5 | off | 11.0 | 13.5 | 30.0 | 7 of 8 | 0 |
| 5 | owner | 11.0 | 27.5 | **92.0** | **2 of 8** | 819 |
| 5 | direct / history | 11.0 | 13.5 | 30.0 | 7 of 8 | 0 |
| 5 | off + cue | 45.0 | 12.0 | 73.0 | 1 of 8 | 0 |

Odour nursery (upstream `odour_nursery.py`, protocol odour-nursery/3), seeds 0-19, lives passing
every check:

| arm | exposure | off | owner | direct / history |
| --- | --- | --- | --- | --- |
| live | 300 | 14 of 20 (lag 56.5) | 14 (54.0) | 14 (56.5), 0 writes |
| live | 1000 | 19 of 20 (lag 40.5) | 20 (40.5) | 19 (40.5), 0 writes |
| step | 300 | 10 of 20 | **3 of 20** | 10, 0 writes |
| step | 1000 | 11 of 20 | **3 of 20** | 11, 0 writes |

Prerequisite chain (upstream `prerequisite_chain.py`, one continuing life of 20,000 moments,
four steps paid only at the end, or shaped), seeds 0-5. Completions per 1,000 moments lived;
greedy and sampled completions per 1,000 of a saved copy at the end:

| mode | pay | off | owner | direct | history |
| --- | --- | --- | --- | --- | --- |
| observed | end | 73.7 / 151.5 / 78.2 | 79.2 / 157.8 / 73.3 | identical to off, 0 writes | identical to off, 0 writes |
| observed | shaped | 114.5 / 160.3 / 67.2 | 112.3 / **103.3** / 62.3 | identical to off, 0 writes | identical to off, 0 writes |
| hidden | end | 35.1 / 0 / 36.8 | 38.0 / 0 / 20.8 | 38.0 / 0 / 20.8 | 36.9 / 0 / 28.5 |
| hidden | shaped | 44.9 / 0 / 33.8 | 46.8 / 0 / 38.3 | 46.8 / 0 / 38.3 | 46.6 / 0 / 43.7 |

The hidden chain is the one place `direct` writes in this panel: the observation is the same at
every stage, so the paid last action meets zero outcomes at the other stages. No arm completes
the hidden chain greedily; completions lived go up slightly under every rule, the sampled policy
at the end goes down with end pay and up with shaped pay (6 seeds, noisy). `history`, which only
writes the history-keyed record, sits between.

## Fresh-seed confirmation

Declared before running (`runs/issue210/run_confirm.sh`): seeds no rule was developed on,
`off` / `owner` / `lib` / `history`. `off` here is the fix copy with the gene at 0, which is
0.89.0 exactly (checked: same lives, same records, and the test suite below).

| check | off (0.89.0) | owner | **lib** | history |
| --- | --- | --- | --- | --- |
| #210 bandit, fail 0: B acquired (24 seeds) | **4/24** | 22/24 | **22/24** | 16/24 |
| same, return to A | 23/24 | 23/24 | 23/24 | 24/24 |
| bandit, fail -1 | 24/24 | 24/24 | 24/24, identical | 24/24, identical |
| songs (reconstruction), mean notes right (gate 0.590) | 0.573 | **0.536** | 0.573, identical, 0 writes | 0.573, identical |
| key-door delay 2: acquire / adapt / return, returns <= 50 (10 seeds) | 6 / 4 / 29, 8 of 10 | 6 / 12.5 / 37.5, 7 of 10 | identical, 0 writes | identical |
| key-door delay 5 | 13 / 10.5 / 46, 5 of 10 | 25 / 32 / **117**, **1 of 10** | identical, 0 writes | identical |
| odour nursery live 300 / 1000: lives passing (20 seeds) | 18, 20 | 18, 20 | identical, 0 writes | identical |
| odour nursery step 300 / 1000 | 17, 12 | **4, 2** | identical, 0 writes | identical |
| observed chain, end: lived / greedy / sampled (6 seeds) | 65.9 / 134.7 / 69.0 | 68.1 / 186.0 / 72.5 | identical, 0 writes | identical |
| observed chain, shaped | 101.6 / 131.8 / 51.8 | 123.2 / 162.0 / 52.2 | identical, 0 writes | identical |
| hidden chain, end | 35.2 / 0 / 30.5 | 39.6 / 0 / 27.2 | 39.6 / 0 / 27.2 | 38.0 / 0 / 31.0 |
| hidden chain, shaped | 42.1 / 0 / 37.2 | 45.9 / 0.3 / 36.2 | 45.9 / 0.3 / 36.2 | 45.3 / 0 / 43.2 |

The songs reconstruction does not pass its own gate on these seeds even for 0.89.0 (0.573 against
0.590); what it shows is that `lib` leaves it exactly as it is while the owner's rule lowers it.

On these seeds the owner's rule did not hurt the observed chain (it helped), unlike the
development seeds (shaped greedy 160 -> 103) and the owner's own panel.

**The 0.9 threshold.** The confirmation first ran the library gene with a payment flag read at
>= 0.5. On the observed chain it then wrote about 90 zeros per condition where the exact-key
prototype wrote none (`confirm_chain_direct.json`): the store's keys overlap, so a payment at one
stage read as about half a payment at the neighbouring stages. A payment is written whole (the
store consolidates at rate 1), so the flag now needs >= 0.9. Rerun after the change: observed
chain identical to `off` with 0 writes (`confirm_chain_lib09_observed.json`), hidden chain
unchanged (`confirm_chain_lib09_hidden.json`, there the payments are real), bandit unchanged
22/24 and 11/12 (`confirm_bandit_lib09.json`, `dev_lib09.json`). The other checks wrote nothing at
0.5, and a stricter flag only removes writes, so they stand. The table above shows `lib` at 0.9.

## The library change

`patches/issue210-memory-extinction.patch` (two commits on a local branch of a separate clone,
`P:/projects/cadence-210fix`, not pushed): `Brain.compose(..., memory_choice=True,
memory_extinction=theta)`, founder 0. `EpisodicRecords` keeps a persistent store of direct
payments (written whole for every known outcome with reward > 0, keyed and separated as the
place records); `outcome` writes a learned zero as a known outcome when the vote for the chosen
action is >= theta and that store reads >= 0.9 for it. Checkpoints keep the store
(`memories/episodic/paid_memory/...`) and refuse one without the other; a brain without the
gene saves exactly as 0.89.0. Docstring and CHANGELOG entry included.

Tests: six new ones (`tests/test_episodic_extinction.py`: validation, gene-off identity, the
#210 bandit 6/6 vs released < 6, a taught act survives a zero outcome, a paid act is
extinguished, a saved brain with the gene continues exactly). Full suite on the fix copy
1947 passed, 80 skipped, 1 xfailed; on the upstream clone 1941 passed, 80 skipped, 1 xfailed
(the same tests plus the six). ruff and mypy clean.

## Reading

- **The cue is not a general workaround.** On the issue's single reversal it is mixed (4/12 vs
  5/12 on development seeds, 14/24 vs 4/24 on fresh ones), and it makes the key-door much slower
  to adapt and return (delay 5: 1 of 8 returns within 50 trips vs 7 of 8). It changes what the
  brain sees, so it changes every task.
- **`direct` / `lib` fixes #210 without touching the chambers the owner's gene broke.** #210
  bandit 4/24 -> 22/24 on fresh seeds (11/12 on development seeds). On songs, key-door, odour
  nursery (live and step) and the observed chain it writes nothing: the lives are identical to
  0.89.0 to the last trip, on development and fresh seeds.
- **The open risk is aliasing**: a directly paid action whose observation hides when it pays (the
  hidden chain). There `lib` writes the same zeros as the owner's gene: completions lived go up
  a little, the sampled policy at the end moves by a few completions either way, no arm completes
  it greedily. `history` (context records only) is gentler there but fixes the bandit less
  (16/24).

## Limits

Songs is a reconstruction, so its numbers are not the owner's (control 0.50 here, 0.54 there).
Small seed counts (6 to 24 per cell). One theta (0.5). The `direct` prototype keys its payment
flags on exact observation codes; the library gene keys them in an associative store (same
results on the bandit, songs and, with the 0.9 flag, the chains, checked). Taught recall, the delayed response, the crafting
chains taught and alone, and V5 corrective were not rerun.

## v2 (2026-10-28, overnight): the remaining checks, bigger panels, a theta sweep

Declared before running in `runs/issue210/run_v2.sh`, `run_v2b.sh`,
`experiments/issue210/checks2.py` and `experiments/issue210/serial_gene.py`. Rules: `off`
(0.89.0), `owner` (his gene as described, rebuilt), `lib` (`memory_extinction`, flag 0.9).

### The remaining in-repo checks

| check (from the repository's tests) | seeds | off | owner | lib |
| --- | --- | --- | --- | --- |
| delayed response from reward (hit rate) | 8-19 | 1.000 | identical, 0 writes | identical, 0 writes |
| continuing contextual bandit, `actor_gamma=0` (hit rate) | 0-23 | 0.946 | identical, 0 writes | identical, 0 writes |
| taught cue recall (`test_the_temporal_context_recalls_a_cue_that_is_gone`, 3 lessons) | 0-23 | 1.000 | identical, 0 writes | identical, 0 writes |

These match the owner's own report for his gene ("identical lives"). Taught cue recall is at
ceiling, so it is a weak check. The crafting chains (taught and alone) and V5 corrective could
not be run: they are not in the repository, like the songs chamber.

### Bigger fresh-seed panel (theta 0.5)

| check | off (0.89.0) | owner | lib |
| --- | --- | --- | --- |
| #210 bandit, fail 0: B acquired (48 seeds, 200-247) | **19/48** | 44/48 | **44/48** |
| same, fail -1 | 48/48 | identical | identical |
| songs (reconstruction, 48 seeds; gate 0.639) | 0.630 | **0.568** | 0.630, identical, 0 writes |
| key-door delay 2: acquire / adapt / return; returns <= 50 (20 seeds) | 6.5 / 9.0 / 37.0; 14 of 20 | 7.0 / 11.5 / 48.5; 10 of 20 | identical, 0 writes |
| key-door delay 5 | 15.0 / 17.0 / 52.5; 10 of 20 | 17.0 / 36.0 / **124.0**; **4 of 20** | identical, 0 writes |
| odour nursery `live` 300 / 1000: lives passing (40 seeds) | 38, 34 | 37, 34 | identical, 0 writes |
| odour nursery `step` 300 / 1000 | 20, 23 | **4, 5** | identical, 0 writes |

The songs reconstruction again misses its own gate even for 0.89.0 (0.630 vs 0.639); what
matters is that `lib` leaves it exactly as it is and the owner's rule lowers it.

### Theta sweep (lib)

| check | 0.25 | 0.5 | 0.75 |
| --- | --- | --- | --- |
| #210 bandit, fail 0: B acquired (48 seeds) | **47/48** (619 writes) | 44/48 (269) | 35/48 (80) |
| songs, key-door, odour nursery, bandit fail -1, observed chain | identical to 0.89.0, 0 writes | identical, 0 writes | identical, 0 writes |
| hidden chain, end pay: lived / sampled at end (seeds 300-305; off 36.2 / 36.3) | 40.2 / 22.8 (8,979 writes) | 39.8 / 28.8 (4,827) | 38.5 / 31.8 (2,543) |
| hidden chain, shaped pay (off 42.9 / 35.2) | 45.6 / 32.3 | 46.5 / 41.5 | 45.9 / 43.3 |

Theta only moves the two places where the gene writes at all: the bandit (lower theta, more
lives fixed) and the hidden chain (lower theta, more writes; the sampled policy at the end drops
with end pay). Everywhere else every theta is byte-for-byte 0.89.0. 0.5 stays a reasonable
middle; 0.25 fixes the bandit almost completely at a larger cost on the aliased chain.

### Aliasing: a context-keyed flag does not help (negative)

`libctx` (rules.py, a prototype): `lib` plus a second condition, that the action was also paid
after a similar history (cosine >= 0.9 between the decision's context key and a context key where
it paid). Declared keep criterion: bandit within 2 lives of `lib`, hidden-chain writes down by at
least half. Result: the bandit is unchanged (77/84 over seeds 0-11, 100-123, 200-247, same as
`lib`), and the hidden chain's writes are **exactly** `lib`'s on all 12 seeds (end 4,786 + 4,827,
shaped 5,498 + 5,598). A diagnostic on seed 100 shows why: at the 812 candidate writes, the context
key's best match to a paid context is 0.983 to 0.9998 (median 0.9994). The temporal records'
key barely separates the chain's stages, so no threshold on it is a principled fix. Not kept,
nothing added to the library. The aliasing case stays open; it needs a key that does carry the
stage (the owner's #189/#211 questions).

### Serial reversal with the gene (failures pay 0, 30 flips, 12 seeds)

| arm | flip 1 | flip 10 | flips 26-30 | slope < 0 |
| --- | --- | --- | --- | --- |
| memory_choice (0.89.0) | 18.1 | 14.7 | 14.4 | 5/12 |
| memory_choice + gene (0.5) | 17.6 | 13.2 | **7.8** | **12/12** |
| cue_choice (cue workaround) | 17.2 | 2.2 | 2.0 | 12/12 |

The 0.89.0 and cue rows reproduce `notes/serial_reversal.md` exactly. With the gene the
learning-set curve comes back in every seed (errors per flip roughly halve over the life), but
it stays well above the cue arm: the gene makes reversals possible, not fast.

### Summary of v2

- No new place where `lib` writes: on every check found in the repository and on bigger panels
  (48 / 20 / 40 seeds) it is identical to 0.89.0 except on the #210 bandit (fixed, 19/48 -> 44/48)
  and the hidden chain (writes, small mixed effects).
- The owner's rule reproduces his reported costs on the bigger panels (key-door delay 5, odour
  nursery through `step`, songs).
- Still not checked: crafting chains, V5 corrective, the original songs chamber (not public).
