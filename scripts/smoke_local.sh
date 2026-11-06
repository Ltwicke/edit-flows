#!/bin/bash
# Local CPU check (Linux / macOS): tests, placeholder data, a two-epoch smoke run and evaluate.py on its
# checkpoint. Run inside the activated environment:
#   scripts/smoke_local.sh
set -euo pipefail
export PYTHONIOENCODING=utf-8
cd "$(dirname "$0")/.."

python -m pytest -q tests -m "not slow"
[ -f data/synthetic/targets.xlsx ] || python scripts/make_synthetic_targets.py --n 20000 --out data/synthetic/targets.xlsx
python train.py experiment=deletion-smoke from_ckpt=none trainer.accelerator=cpu
python evaluate.py runs/smoke/checkpoints/last.ckpt -n 32 -s 20 --show 2
