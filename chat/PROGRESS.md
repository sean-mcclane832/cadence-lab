# Cadence chat: progress log

Goal (Sean, 2026-10-28): train a Cadence brain to form sentences and hold a (simple)
conversation. Library: cadence-net **0.91.1** (installed from the upstream clone at c9c8080)
unless a section says otherwise. Per Sean's direction the final result is **one brain**
trained as long as possible, not a seed average; baselines are one run each.

## Design (stage 0)

**Corpus** ([corpus.py](corpus.py)). Self-written template grammar of two-line exchanges,
`you: <prompt>\nbot: <reply>\n`, 16 intents (greetings, how are you, names, likes, pets,
places, thanks, bye...), 34-character alphabet. 6,000 training exchanges (277k chars), fresh
validation and test draws from other seeds. Every prompt comes with its full set of valid
replies, so a generated reply is scored **valid** or not. The grammar has only 275 distinct
prompts, so 99% of test exchanges also occur in training: this tests whether the brain can
learn and reproduce a small conversational repertoire, not open-ended generalisation. A
separate **novel** set uses names and things never seen in training ("my name is zoe" ->
"nice to meet you, zoe."), which can only be answered by copying from the prompt.

**Text in.** One character per moment. The observation is the one-hot code of the last
`window` characters (window 1 = only the current char). Context beyond that comes from
Cadence's own memories: the default working trace (a copy of last moment's association
activity) and, in most configs, the opt-in heard history (`history_decays=(0.2, 0.8, 0.95)`:
fading copies of the input at three timescales).

**Text out.** One action slot of 34 choices; the action issued after reading char t is the
brain's guess at char t+1. To write a reply the brain reads `you: ...\nbot: `, then each
issued action is fed back as the next observation until it issues a newline.

**Learning signal.** Supervised teaching: `Brain.step(x_t, teacher=[c_{t+1}])`, the
composed teaching learner (free and nudged settle, local contrast), with `learning_surprise=True`
(a lesson only when its own act was wrong). Reward rates are zero; the text pays nothing.
This is the same recipe upstream uses in its streaming-words chamber. One continuing stream,
no resets during training.

**Readings.** Next-char accuracy (greedy) and bits per char (softmax temperature picked on
validation, separately for each model), accuracy on reply characters only, valid-reply rate
in a continuing conversation (the model hears its own replies), and the novel-copy rate.
Baselines with the same harness ([baselines.py](baselines.py)): Witten-Bell char n-gram,
window-3 MLP, GRU.

Upstream's own guidance applies to how far any of this goes: "Next-event accuracy and
self-feeding alone are bounded component results, not evidence of phrase or whole-sequence
competence" (docs/task-design.md).

## Stage 1: characters (screening, 2026-10-29 01:45-03:10)

One brain per config, seed 0, one continuing stream over the training text, probes on 120
validation exchanges (5.5k chars). These are screening runs to choose a recipe, not results.

| Config (all `learning_surprise=True`) | params | chars seen | val char acc | val bpc | valid replies |
| --- | --- | --- | --- | --- | --- |
| `plain`: founder, window 1, working trace only | 9.9k | 60k | 0.17 (train acc falling 0.09 -> 0.05, stopped) | 4.59 | 0 |
| `hist`: + heard history (0.2, 0.8, 0.95) | 16.5k | 150k | 0.21 / 0.41 / 0.32 / 0.14 / 0.35 | 3.32 | 0 |
| `win3`: last 3 chars + history | 36.5k | 150k | 0.42 / 0.34 / 0.39 / 0.07 / 0.26 | 3.86 | 0 |
| `win3_norm`: + `teaching="normalized"` | 36.5k | 150k | 0.64 / 0.59 / 0.61 / 0.65 / **0.71** | **1.58** | 0 |
| `win5_m256_norm`: 5 chars, 256-unit region | ~150k | 30k | 0.63 | 2.08 | 0 (stopped: 5x slower, no better) |
| `win8_m256_norm`: 8 chars, 256-unit region | ~250k | 30k | 0.62 | 2.26 | 0 (stopped) |

Reference on the same text (test split, one run each): char n-gram order 2 (bigram) 0.40 acc
/ 2.67 bpc; order 4 0.80 / 0.76; order 8 0.90 / 0.35; **order 12 0.92 / 0.25, 47% valid
replies**. Even the n-gram needs ~12 chars of context before replies start to make sense:
an order-8 model cannot tell `bot: hello ` from `you: hello ` and drifts into prompts.

What I learned:
- **The composed founder's teaching point is unstable on text.** Its validation accuracy
  swings by 0.3 between probes and never beats a bigram. This matches upstream's own note
  that with many lessons the founder rates saturate; `teaching="normalized"` (their
  recommended point for imitation) fixed it here: steady climb to 0.71.
- **A pressed-in window beats Cadence's own memories.** One char in + working trace + heard
  history reaches ~0.4 at best; giving it the last 3 chars directly gets 0.64 within 30k.
- **Bigger regions were not better per char seen** at 30k and cost 5x the time (~35 chars/s
  vs ~170), so they were dropped.
- The char brain writes English-shaped fragments but no valid reply yet:
  `how are you?` -> `no, nikk thall.`; `my pet is a rabbit` -> `hello, i like ti to theth.!you do?`;
  `what is your favorite color?` -> `i like b ritt ththp pizza.`

`win3_norm` continues from its 150k checkpoint in `runs/long_char/` (one core).

Corpus quirk, kept so results stay comparable: the pet reply `i like {a}s.` makes "fishs".

## Stage 2: word tokens (2026-10-29 02:00-02:45)

A reply depends on the whole prompt, ~25 chars back but only ~6 words back. So the second
design reads and writes **one word token per moment** ([words.py](words.py),
[wordbrain.py](wordbrain.py)): 188 tokens (words, `? ! . ,`, `you:`, `bot:`, `<eol>`),
82k training tokens (13.7 per exchange). Observation: one-hot of the last 8 tokens (1,504
values). One action slot over the 188 tokens.

Three recipe changes, each measured:

1. **Heard history dropped for words.** It copies the whole windowed observation at each
   decay (4,828 neurons, 522k params, ~15 tokens/s). The window already carries the context.
2. **Supervised-only loop + no associative store** (`lesson(x, y)` then a greedy
   `act(x)`, `episodic=False`), upstream's recipe for "demonstrations with no reward
   protocol". A profile showed a third of each `step` in the associative store recording
   zero-reward outcomes and more in the reward learner at zero rates. 28 -> 75 tokens/s.
3. **16 parallel streams per lesson** (`streams=16`: 16 evenly spaced read positions in
   the text, one lesson teaches all 16 rows). Upstream specifies its normalized teaching
   point at ~32 rows per lesson. This was the biggest win, in both speed and quality.

| Config (window 8, normalized, surprise) | params | tokens | val token acc (probes) | valid replies |
| --- | --- | --- | --- | --- |
| `w6_nohist`, `step` loop, 1 stream | 302k | 20k | 0.37, 0.44 | 0.05 |
| `w8_nohist`, `step` loop, 1 stream | ~360k | 20k | 0.52, 0.58 | 0.01 |
| `s8`, lesson loop, no store, 1 stream | 114k | 80k | 0.48, 0.51, 0.53, 0.50, 0.44, 0.45 | 0.06-0.23, swinging |
| `s8_m128`, 128-unit region | 235k | 10k | **collapsed**: 0.004, every reply "emma emma emma..." | 0 |
| `s8_b32`, 32 streams | 114k | 100k | 0.54, 0.48, 0.66, 0.53, 0.61 | 0.33 |
| **`s8_b16`, 16 streams** | 114k | 100k | 0.62, 0.63, 0.71, 0.75, **0.75** | **0.30** |

Word baselines (test split, one run each): word n-gram order 6 0.80 acc / 70% valid,
**order 8 0.82 / 86% valid**; GRU (hidden 64, 61k params, 400k tokens) 0.80 / 59% valid.
(0.82 is close to the ceiling: many replies are a free choice among three.)

`s8_b16` samples at 100k tokens (validation conversation, greedy, brain hears its own replies):

```
you: what is your name?          bot: i am cadence, a small brain.
you: hello                       bot: hi, nice to meet you.
you: what is your favorite color? bot: i like blue.
you: my pet is a rabbit          bot: a snake? that is cool.
you: how are you?                bot: i am a bot.
you: goodbye                     bot: hi,
```

Whole, grammatical sentences from the training repertoire, often the wrong one for the
prompt (greetings/byes/thanks at 0%). It has learned the replies better than the mapping
from prompt to reply. `s8_b16` continues as the main brain from its 100k checkpoint
(`runs/main/`).

## Stage 3: the collapse, and what it actually was (02:45-03:25)

`s8_b16` continued from 100k as the main brain **collapsed**: val token accuracy 0.75 ->
0.12 -> 0.0006 within 1.2 more epochs, train accuracy 0.03, lessons on 97% of rows. Two
tuning arms (128-unit region, 12-token window) went the same way. I missed the warning: the
train accuracy had already dipped at the 100k probe.

Tuning arms, all from the healthy 100k checkpoint (val acc per 20k probe):

| Arm | val token acc | reading |
| --- | --- | --- |
| eta 0.003 (the normalized point) | 0.12, 0.13, 0.0006 | weights broken: cold settle also 0.15 |
| eta 0.001 | 0.72, **0.17, 0.18** | looked collapsed, but see below |
| eta 0.001, `scale_cap` 2 | 0.72, 0.76, 0.76, 0.77, 0.50 | going |
| eta 0.0003 | 0.75, 0.75, 0.75, 0.76, 0.77, 0.75, 0.75, **0.48, 0.48**, 0.75 | stable weights, flipping probes |
| eta 0.0003 + reset streams every 1,024 steps | 0.74, 0.75, 0.74 (train acc 0.74 vs 0.48 without) | |

Two diagnostics explained it:

- **No drift in a frozen brain.** One stream, 8,000 tokens, no learning: accuracy per
  500-token bin stays at 0.74-0.78 the whole way, with or without resets ([drift.py](drift.py)).
- **The warm start traps streams.** Each settle starts from the stream's last settled state.
  The eta 0.001 brain at 160k scores 0.18 warm, **0.77 when every settle starts from rest**
  (working trace and weights untouched), 0.75 when reset at each `you:`. Its weights were
  fine; its stream had fallen into a bad settled state that the warm start kept it in. The
  flipping 0.75/0.48 probes of the eta 0.0003 brain are the same thing. On the healthy brain
  cold settles are also better (0.775 vs 0.755). The real weight collapse (eta 0.003) stays
  broken cold (0.15).

This is the opposite of the Breakout settle probe, where removing the warm start cost almost
nothing. On text, the warm start is a liability. There is no public switch for it, so the
brain is run with `basal_ganglia._free = None` before each settle (the settle probe's
"cold" lever, private API, documented in `wordbrain.cold`).

Cold settles during training, from the eta 0.0003 brain at 300k: train acc 0.48 -> 0.79,
val 0.78, bits per token 2.0 -> **1.17**, valid replies 0.52 in the first 3k tokens. That is the
recipe for the main brain from here on: `main_cold` = window 8, 16 streams, normalized
teaching at eta 0.0003, surprise lessons, resets every 1,024 steps, cold settles, no
associative store. One brain, continued from the eta 0.0003 brain's 300k checkpoint, in
`runs/main_cold/`.

Char baseline added: GRU (hidden 64, 1M chars) 0.91 char accuracy, 0.40 bpc, 42% valid replies.
