# Cadence chat

Can a Cadence brain (cadence-net 0.91.1) learn to form sentences and answer typed lines?
A small, honest attempt. Results, numbers and samples: [PROGRESS.md](PROGRESS.md).

## Run it

From this folder, on Sean's PC (the venv already exists):

```bash
.venv/Scripts/python chat.py                     # talk to the shipped brain
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

Brain files (`*.npz`) are ignored by git, so a fresh clone has no brain to talk to: train one
first (below) and point `--brain` at it.

## Files

| File | What |
| --- | --- |
| `corpus.py` | the self-written chat grammar (16 intents), splits, valid-reply checker, novel-word set |
| `textbrain.py`, `train_cadence.py` | character-level Cadence brain (stage 1) |
| `words.py`, `wordbrain.py` | word-token Cadence brain (stage 2, the main line) |
| `baselines.py` | char/word n-gram, window MLP, GRU on the same scoring |
| `chat.py` | REPL over a saved word brain |
| `runs/**/log.jsonl` | one JSON line per probe for every run |

## Train

```bash
.venv/Scripts/python wordbrain.py --cfg s8 --tokens 200000 --probe 20000 --out runs/my_s8
.venv/Scripts/python wordbrain.py --cfg s8 --tokens 200000 --resume runs/my_s8/brain.npz --start 200000 --out runs/my_s8
.venv/Scripts/python baselines.py ngram --words
.venv/Scripts/python baselines.py gru --words --hidden 64 --chars 400000
```
