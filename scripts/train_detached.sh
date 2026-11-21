#!/bin/bash
# Start a training run on a GPU node that has NO batch scheduler, detached, so it survives logout.
# On a cluster with SLURM use scripts/train.sbatch instead.
#
#   scripts/train_detached.sh <run name> [hydra overrides...]
#
#   EDITFLOW_EXPERIMENT=deletion-gpu-check scripts/train_detached.sh gpu-check
#   scripts/train_detached.sh long-01 trainer.max_epochs=8
#   EDITFLOW_GPUS=0 scripts/train_detached.sh long-01            # one GPU of several
#
# Environment:
#   EDITFLOW_EXPERIMENT  experiment config          (default deletion-gpu-24h)
#   EDITFLOW_VENV        virtual environment        (default $HOME/0envs/EditFlows_env)
#   EDITFLOW_GPUS        comma-separated indices    (default: every GPU on the node)
#   EDITFLOW_WORKERS     dataloader workers / rank  (default: cores per rank - 1, capped at 8)
#   EDITFLOW_FOREGROUND  1 = run in the foreground instead of detaching
#   EDITFLOW_FORCE       1 = start even though the GPUs are in use by someone else
#
# Without a scheduler there is no admission control, so this refuses to start on a GPU that already has
# processes on it: two jobs on one card means both run out of memory. Re-running the same run name
# continues from its newest checkpoint, exactly as under SLURM - resuming is checkpoint-based.

set -euo pipefail

RUN_NAME="${1:?usage: scripts/train_detached.sh <run name> [hydra overrides...]}"
shift

EXPERIMENT="${EDITFLOW_EXPERIMENT:-deletion-gpu-24h}"
VENV="${EDITFLOW_VENV:-$HOME/0envs/EditFlows_env}"
PYTHON="$VENV/bin/python"

cd "$(dirname "$0")/.."

[ -x "$PYTHON" ] || { echo "no interpreter at $PYTHON - build the environment or set EDITFLOW_VENV" >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found - is this a GPU node?" >&2; exit 1; }

# Which GPUs?
if [ -n "${EDITFLOW_GPUS:-}" ]; then
    GPUS="$EDITFLOW_GPUS"
else
    GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
fi
[ -n "$GPUS" ] || { echo "no GPUs found on $(hostname)" >&2; exit 1; }
N_GPUS="$(printf '%s' "$GPUS" | tr ',' '\n' | grep -c .)"

# Refuse to share a card, unless told otherwise.
BUSY=""
for idx in $(printf '%s' "$GPUS" | tr ',' ' '); do
    procs="$(nvidia-smi -i "$idx" --query-compute-apps=pid,process_name,used_memory \
             --format=csv,noheader 2>/dev/null || true)"
    [ -n "$procs" ] && BUSY="${BUSY}  GPU ${idx}: ${procs}"$'\n'
done
if [ -n "$BUSY" ]; then
    echo "GPUs already in use on $(hostname):" >&2
    printf '%s' "$BUSY" >&2
    if [ "${EDITFLOW_FORCE:-0}" != "1" ]; then
        echo "refusing to start; pick free GPUs with EDITFLOW_GPUS=... or set EDITFLOW_FORCE=1" >&2
        exit 1
    fi
    echo "EDITFLOW_FORCE=1 - starting anyway" >&2
fi

CPUS="$(nproc)"
N_WORKERS="${EDITFLOW_WORKERS:-$(( CPUS / N_GPUS - 1 ))}"
[ "$N_WORKERS" -lt 0 ] && N_WORKERS=0
[ "$N_WORKERS" -gt 8 ] && N_WORKERS=8

mkdir -p runs
PID_FILE="runs/${RUN_NAME}.pid"
LOG_FILE="runs/${RUN_NAME}.detached.out"

# Two processes writing one run directory would corrupt it.
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "run '$RUN_NAME' is already running as pid $(cat "$PID_FILE")" >&2
    echo "stop it with: kill \$(cat $PID_FILE)" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPUS"

echo "host        $(hostname)"
echo "experiment  $EXPERIMENT"
echo "run         $RUN_NAME   ->  runs/$RUN_NAME"
echo "gpus        $GPUS  ($N_GPUS visible)   workers/rank $N_WORKERS of $CPUS cores"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv

set -- "$PYTHON" train.py \
    experiment="$EXPERIMENT" \
    name="$RUN_NAME" \
    gpus="$N_GPUS" \
    data.num_workers="$N_WORKERS" \
    "$@"

if [ "${EDITFLOW_FOREGROUND:-0}" = "1" ]; then
    exec "$@"
fi

# New session, no controlling terminal, stdin closed: survives logout.
setsid nohup "$@" > "$LOG_FILE" 2>&1 < /dev/null &
echo $! > "$PID_FILE"

echo
echo "started pid $(cat "$PID_FILE")"
echo "  tail -f runs/$RUN_NAME/train.log      # progress and the per-epoch deletion metrics"
echo "  tail -f $LOG_FILE   # stdout/stderr, including any crash"
echo "  nvidia-smi"
echo "  kill \$(cat $PID_FILE)                 # stop it; resume by running this script again"
