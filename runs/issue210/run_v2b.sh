#!/bin/bash
# v2b, declared 2026-10-28 ~00:45 before running: the aliasing prototype `libctx` (rules.py;
# the library gene plus a context flag, cosine match >= 0.9 to a context key where this action
# was paid). Its writes are a subset of lib's by construction (it only removes writes), so on
# the chambers where lib writes nothing (songs, key-door, odour nursery, observed chain, the
# in-repo checks) it is identical to lib and to 0.89.0 and is not rerun there.
# Run: the #210 bandit on the development seeds 0-11, the fresh seeds 100-123 and the big
# panel's 200-247 (compare lib: 11/12, 22/24 and v2), and the hidden chain on seeds 100-105
# (compare confirm_chain.json off and confirm_chain_lib09_hidden.json lib) and 300-305
# (compare v2 theta0.5_chain.json).
# Keep libctx if: the bandit stays within 2 lives of lib on each seed set, and the hidden
# chain's writes fall by at least half against lib.
cd P:/projects/cadence-lab
PY=P:/projects/cadence/.venv/Scripts/python.exe
export PYTHONUTF8=1
export PYTHONPATH=P:/projects/cadence-210fix/src
R=runs/issue210/v2
W=6
$PY experiments/issue210/chambers.py --only bandit --seeds $(seq 0 11) $(seq 100 123) $(seq 200 247) --rules libctx --workers $W --out $R/libctx_bandit.json > $R/libctx_bandit.log 2>&1
$PY experiments/issue210/summarize.py $R/libctx_bandit.json >> $R/libctx_bandit.log 2>&1
$PY experiments/issue210/chain.py --modes hidden --seeds $(seq 100 105) $(seq 300 305) --rules libctx --workers $W --out $R/libctx_chain.json > $R/libctx_chain.log 2>&1
echo V2B_DONE > $R/DONE_B
