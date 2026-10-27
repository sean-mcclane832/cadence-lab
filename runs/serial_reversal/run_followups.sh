#!/bin/bash
# Follow-ups, 2026-10-27 ~01:00: probe block, issue #210 zero-payoff case, naive controls,
# then rebuild the earlier receipts on the final source (defaults unchanged).
cd P:/projects/cadence-lab
PY=P:/projects/cadence/.venv/Scripts/python.exe
export PYTHONUTF8=1
S="0 1 2 3 4 5 6 7 8 9 10 11"
D=runs/serial_reversal
$PY experiments/serial_reversal.py --seeds $S --probe 400 --out $D/receipt_probe.json > $D/probe.log 2>&1
$PY experiments/serial_reversal.py --seeds $S --reversals 1 --first 400 --out $D/receipt_naive400.json > $D/naive400.log 2>&1
$PY experiments/serial_reversal.py --seeds $S --fail 0 --out $D/receipt_fail0.json > $D/fail0.log 2>&1
$PY experiments/serial_reversal.py --seeds $S --fail 0 --reversals 30 --arms continuing memory_choice cue cue_choice mlp_cbp wsls --out $D/receipt_fail0_long.json > $D/fail0_long.log 2>&1
$PY experiments/serial_reversal.py --seeds $S --out $D/receipt.json > $D/main.log 2>&1
$PY experiments/serial_reversal.py --seeds $S --first 100 --out $D/receipt_equal.json > $D/equal.log 2>&1
$PY experiments/serial_reversal.py --seeds $S --p 0.9 --out $D/receipt_p0.9.json > $D/p0.9.log 2>&1
$PY experiments/serial_reversal.py --seeds $S --reversals 30 --out $D/receipt_long.json > $D/long.log 2>&1
echo ALL_DONE
