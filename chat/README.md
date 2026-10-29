# Cadence chat

Can a Cadence brain (cadence-net 0.91.1) learn to form sentences and answer typed lines?
A small, honest attempt. Results, numbers and samples: [PROGRESS.md](PROGRESS.md).

## Run it

From this folder, on Sean's PC (the venv already exists):

```bash
.venv/Scripts/python chat.py                     # talk to the best brain (brains/copy, local only)
.venv/Scripts/python chat.py --brain brains/chat/brain.npz   # the smaller brain that is in git
.venv/Scripts/python chat.py --temperature 0.3   # sample replies instead of greedy
.venv/Scripts/python chat.py --say "hi" "my name is sam" "do you like pizza?"   # scripted
```

Type lowercase lines like the ones in the corpus (`hi`, `how are you?`, `i have a dog`,
`do you like chess?`, `what is your name?`, `bye`). Words it never saw are read as unknown
input and it cannot say them back. It does not learn while you talk.

Fresh setup elsewhere:

```bash
uv venv -p 3.11 .venv
uv pip install -p .venv "cadence-net[fast]==0.91.1" numpy   # or the upstream clone at 0.91.1
uv pip install -p .venv torch --index-url https://download.pytorch.org/whl/cpu   # baselines only
```

Two trained brains (one life each, not seed averages):

- `brains/copy/` (copy8: 8-word window + recall memory with upstream's copy pathway, 3.6M
  parameters, 680k tokens): the best. 83% valid replies on the held-out test. 45 MB, so it
  is **not in git**; on Sean's PC it is the default.
- `brains/chat/` (main_cold_anneal: 8-word window, 64-unit region, 114k parameters, 1.52M
  tokens): 70% valid. 3.7 MB, **committed**, so a fresh clone can chat with it.

Both run with cold settles (each settle starts from rest, see PROGRESS.md stage 3).

## Files

| File | What |
| --- | --- |
| `corpus.py` | the self-written chat grammar (16 intents), splits, valid-reply checker, novel-word set |
| `textbrain.py`, `train_cadence.py` | character-level Cadence brain (stage 1) |
| `words.py`, `wordbrain.py` | word-token Cadence brain (stage 2, the main line) |
| `baselines.py` | char/word n-gram, window MLP, GRU on the same scoring |
| `chat.py` | REPL over a saved word brain |
| `eval_final.py`, `drift.py` | held-out test with window-cut ablations; frozen-stream drift check |
| `runs/**/log.jsonl` | one JSON line per probe for every run |

## Train

```bash
.venv/Scripts/python wordbrain.py --cfg s8 --tokens 200000 --probe 20000 --out runs/my_s8
.venv/Scripts/python wordbrain.py --cfg s8 --tokens 200000 --resume runs/my_s8/brain.npz --start 200000 --out runs/my_s8
.venv/Scripts/python baselines.py ngram --words
.venv/Scripts/python baselines.py gru --words --hidden 64 --chars 400000
```
