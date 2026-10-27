#!/bin/bash
# Overnight batch, 2026-10-27. Logs next to the receipts.
cd P:/projects/cadence-lab
PY=P:/projects/cadence/.venv/Scripts/python.exe
export PYTHONUTF8=1
S="0 1 2 3 4 5 6 7 8 9 10 11"
$PY experiments/serial_reversal.py --seeds $S --out runs/serial_reversal/receipt.json > runs/serial_reversal/main.log 2>&1
$PY experiments/serial_reversal.py --seeds $S --first 100 --out runs/serial_reversal/receipt_equal.json > runs/serial_reversal/equal.log 2>&1
$PY experiments/serial_reversal.py --seeds $S --p 0.9 --out runs/serial_reversal/receipt_p0.9.json > runs/serial_reversal/p0.9.log 2>&1
$PY experiments/serial_reversal.py --seeds $S --reversals 30 --out runs/serial_reversal/receipt_long.json > runs/serial_reversal/long.log 2>&1
echo ALL_DONE
