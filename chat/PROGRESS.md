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
