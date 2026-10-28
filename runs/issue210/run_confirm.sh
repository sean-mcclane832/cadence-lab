#!/bin/bash
# Fresh-seed confirmation, declared 2026-10-27 before running: the library gene
# (memory_extinction=0.5, cadence-210fix) against 0.89.0 as released and the owner's rule
# (prototype), on seeds no rule was developed on.
cd P:/projects/cadence-lab
PY=P:/projects/cadence/.venv/Scripts/python.exe
export PYTHONUTF8=1
export PYTHONPATH=P:/projects/cadence-210fix/src
R=runs/issue210
$PY experiments/issue210/chambers.py --seeds $(seq 100 123) --rules off owner lib history --workers 7 --out $R/confirm_chambers.json > $R/confirm_chambers.log 2>&1
$PY experiments/issue210/summarize.py $R/confirm_chambers.json >> $R/confirm_chambers.log 2>&1
$PY experiments/issue210/keydoor.py --seeds $(seq 100 109) --delays 2 5 --rules off owner lib history --workers 7 --out $R/confirm_keydoor.json > $R/confirm_keydoor.log 2>&1
$PY experiments/issue210/odour.py --seeds $(seq 500 519) --rules off owner lib history --workers 7 --out $R/confirm_odour.json > $R/confirm_odour.log 2>&1
$PY experiments/issue210/chain.py --seeds $(seq 100 105) --rules off owner lib history --workers 7 --out $R/confirm_chain.json > $R/confirm_chain.log 2>&1
echo CONFIRM_DONE >> $R/confirm_chain.log
