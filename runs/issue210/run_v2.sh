#!/bin/bash
# Overnight v2, declared 2026-10-28 before running (see experiments/issue210/checks2.py and
# notes/issue210_checks.md "v2"):
#  1. the remaining in-repo preservation checks (delayed response, contextual bandit, taught cue
#     recall), off / owner / lib at theta 0.5;
#  2. bigger fresh-seed panel, off / owner / lib at 0.5: bandit + songs 48 seeds (200-247),
#     key-door 20 seeds (200-219), odour nursery 40 seeds (600-639);
#  3. theta sweep of lib at 0.25 and 0.75 on the same seeds (0.5 is in step 2), plus both
#     chains (6 seeds, 300-305; off and lib 0.5 alongside for the same seeds).
# Prediction: lib writes nothing outside the bandit and the hidden chain at every theta; the
# bandit fix holds at 0.25 and 0.5, and is weaker at 0.75 (the failing arm's vote sits near .5).
cd P:/projects/cadence-lab
PY=P:/projects/cadence/.venv/Scripts/python.exe
export PYTHONUTF8=1
export PYTHONPATH=P:/projects/cadence-210fix/src
R=runs/issue210/v2
mkdir -p $R
W=6
$PY experiments/issue210/checks2.py --workers $W --out $R/checks2.json > $R/checks2.log 2>&1
$PY experiments/issue210/chambers.py --seeds $(seq 200 247) --rules off owner lib --workers $W --out $R/big_chambers.json > $R/big_chambers.log 2>&1
$PY experiments/issue210/summarize.py $R/big_chambers.json >> $R/big_chambers.log 2>&1
$PY experiments/issue210/keydoor.py --seeds $(seq 200 219) --delays 2 5 --rules off owner lib --workers $W --out $R/big_keydoor.json > $R/big_keydoor.log 2>&1
$PY experiments/issue210/odour.py --seeds $(seq 600 639) --rules off owner lib --workers $W --out $R/big_odour.json > $R/big_odour.log 2>&1
for T in 0.25 0.75; do
  $PY experiments/issue210/chambers.py --seeds $(seq 200 247) --rules lib --theta $T --workers $W --out $R/theta${T}_chambers.json > $R/theta${T}_chambers.log 2>&1
  $PY experiments/issue210/summarize.py $R/theta${T}_chambers.json >> $R/theta${T}_chambers.log 2>&1
  $PY experiments/issue210/keydoor.py --seeds $(seq 200 219) --delays 2 5 --rules lib --theta $T --workers $W --out $R/theta${T}_keydoor.json > $R/theta${T}_keydoor.log 2>&1
  $PY experiments/issue210/odour.py --seeds $(seq 600 639) --rules lib --theta $T --workers $W --out $R/theta${T}_odour.json > $R/theta${T}_odour.log 2>&1
  $PY experiments/issue210/chain.py --seeds $(seq 300 305) --rules lib --theta $T --workers $W --out $R/theta${T}_chain.json > $R/theta${T}_chain.log 2>&1
done
$PY experiments/issue210/chain.py --seeds $(seq 300 305) --rules off lib --theta 0.5 --workers $W --out $R/theta0.5_chain.json > $R/theta0.5_chain.log 2>&1
echo V2_DONE > $R/DONE
