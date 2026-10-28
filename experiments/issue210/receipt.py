"""Bundle the issue #210 runs into one cadence.Receipt bound to this lab's sources.

Run:  python experiments/issue210/receipt.py          (writes runs/issue210/receipt.json)
      python experiments/issue210/receipt.py --verify
The library change itself is bound through patches/issue210-memory-extinction.patch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cadence as cd

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs" / "issue210"
SOURCES = [
    (f"experiments/issue210/{name}", ROOT / "experiments" / "issue210" / name)
    for name in ("rules.py", "chambers.py", "keydoor.py", "odour.py", "chain.py", "summarize.py",
                 "checks2.py", "serial_gene.py", "mark.py")
] + [("patches/issue210-memory-extinction.patch", ROOT / "patches" / "issue210-memory-extinction.patch")]
RUN_FILES = [
    "dev.json", "dev_history.json", "dev_lib.json", "keydoor_dev.json", "keydoor_dev_cue.json",
    "keydoor_dev_history.json", "odour_dev.json", "odour_dev_history.json", "chain_dev.json",
    "chain_dev_history.json", "confirm_chambers.json", "confirm_keydoor.json",
    "confirm_odour.json", "confirm_chain.json", "dev_lib09.json", "confirm_bandit_lib09.json",
    "confirm_chain_direct.json", "confirm_chain_lib09_observed.json",
    "confirm_chain_lib09_hidden.json",
    # v2, 2026-10-28
    "v2/checks2.json", "v2/big_chambers.json", "v2/big_keydoor.json", "v2/big_odour.json",
    "v2/theta0.25_chambers.json", "v2/theta0.25_keydoor.json", "v2/theta0.25_odour.json",
    "v2/theta0.25_chain.json", "v2/theta0.75_chambers.json", "v2/theta0.75_keydoor.json",
    "v2/theta0.75_odour.json", "v2/theta0.75_chain.json", "v2/theta0.5_chain.json",
    "v2/libctx_bandit.json", "v2/libctx_chain.json", "v2/serial_gene.json", "v2/mark.json",
]


def main() -> None:
    path = RUNS / "receipt.json"
    if "--verify" in sys.argv:
        ok, message = cd.Receipt.verify(path, sources=SOURCES)
        print(ok, message)
        sys.exit(0 if ok else 1)
    body = {
        "schema": "cadence-lab/issue210-checks/v2",
        "cadence": cd.__version__,
        "upstream_commit": "81345d8",
        "runs": {name: json.loads((RUNS / name).read_text(encoding="utf-8"))["rows"] for name in RUN_FILES},
    }
    cd.Receipt.build(body["schema"], body, sources=SOURCES).write(path)
    ok, message = cd.Receipt.verify(path, sources=SOURCES)
    print(path, ok, message)


if __name__ == "__main__":
    main()
