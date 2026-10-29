# cadence-lab

Experiments and simulations with [Cadence](https://github.com/muellerberndt/cadence), the
local-learning equilibrium brain from Pragma Research.

This is an independent hobby lab. It is **not affiliated with or endorsed by** Pragma Research
or the Cadence authors. Results here are small, exploratory and specific to the seeds and
settings stated in each receipt.

## Layout

| Path | What |
| --- | --- |
| `envs/` | Small task environments (bandits, gridworlds) |
| `experiments/` | Runnable experiments, one file each |
| `runs/` | Outputs: receipts (`cadence.Receipt` JSON), logs |
| `notes/` | Short write-ups with result tables |

## Setup

Python 3.11+, Cadence 0.89.0.

```bash
uv venv -p 3.11 .venv
uv pip install -p .venv "cadence-net[fast]==0.89.0" numpy torch
# For an NVIDIA GPU, install the CUDA build of torch from https://download.pytorch.org/whl/
```

On consumer NVIDIA cards use float32 for torch-backend brains (float64 is very slow there).
A composed brain takes its precision on load:
`Brain.load(path, backend="torch", precision="float32")`.

## Experiments

- [`experiments/bandit_reversal.py`](experiments/bandit_reversal.py): two-armed bandit whose
  paying arm moves mid-life. One continuing `Brain.compose(..., arousal=True)` brain against a
  cold reset, a newborn brain, a memory-off brain, an online PyTorch MLP and uniform random.
  Write-up: [`notes/bandit_reversal.md`](notes/bandit_reversal.md).

- [`experiments/backend_timing.py`](experiments/backend_timing.py): the same brain on CPU
  (NumPy + Numba), CUDA float64 and CUDA float32. Write-up: [`notes/backend_timing.md`](notes/backend_timing.md).
- [`experiments/serial_reversal.py`](experiments/serial_reversal.py): the paying arm flips 10
  to 30 times in one life. Does a continuing brain get faster at switching, as animals do?
  Write-up: [`notes/serial_reversal.md`](notes/serial_reversal.md).
- [`experiments/serial_reversal_recheck.py`](experiments/serial_reversal_recheck.py): is that
  speed-up learning to learn? Age, savings, arousal and working-memory controls (pre-registered).
  Partly: about half is savings on previously paid arms. Write-up:
  [`notes/serial_reversal_recheck.md`](notes/serial_reversal_recheck.md).
- [`experiments/issue210/`](experiments/issue210/): the preservation checks of upstream issue
  #210 (zero outcomes never extinguish a paid record) and a narrower extinction rule that keeps
  them. Write-up: [`notes/issue210_checks.md`](notes/issue210_checks.md).
- [`experiments/animal_battery.py`](experiments/animal_battery.py): five classic conditioning
  effects (Kamin blocking, partial reinforcement extinction, ABA renewal, spontaneous recovery,
  latent inhibition) on a continuing Cadence brain and an MLP. Write-up:
  [`notes/animal_battery.md`](notes/animal_battery.md).
- `runs/ladder/`, `runs/demos/`, `runs/baseline/`: outputs from the upstream examples, the
  cadence-demos test suites and the Cadence test suite on Windows 10 (Python 3.11).

```bash
python experiments/bandit_reversal.py --seeds 0 1 2 3 4 5 6 7 8 9 10 11
python experiments/bandit_reversal.py --life 900 --reversal 300 --back 600   # A-B-A
python experiments/backend_timing.py --sizes 64 512 --moments 300
```

## License

GPL-3.0, matching Cadence. See [LICENSE](LICENSE).
