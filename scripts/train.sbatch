#!/bin/bash
# SLURM job template for training on a GPU node. Fill in the placeholders in <...> for your site.
#
# NOTE: the UiO CCSE cluster this fork is used on has NO scheduler installed - there is no sbatch,
# squeue or sinfo there. Use scripts/train_detached.sh instead; see the appendix of docs/CLUSTER.md.
# This template is kept for clusters that do run SLURM.
#
#   mkdir -p runs                                  # ONCE: --output is resolved at submit time
#   sbatch scripts/train.sbatch <run name> [hydra overrides...]
#
#   EDITFLOW_EXPERIMENT=deletion-gpu-check sbatch scripts/train.sbatch gpu-check
#   sbatch scripts/train.sbatch long-01 trainer.max_epochs=8
#   sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/train.sbatch big-01     # 4 GPUs on one node, DDP
#
# sbatch command-line options override the #SBATCH directives below, so the node, partition, GPU count
# and time limit are changeable without editing this file.
#
# A run resumes automatically from its newest checkpoint when the same run name is submitted again, so a
# run that needs more than one job's time limit is simply resubmitted (see scripts/submit_chain.sh).

#SBATCH --job-name=editflow-deletion
#SBATCH --partition=<gpu-partition>
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-23:30:00
#SBATCH --output=runs/slurm-%x-%j.out

# Notes on the directives above:
#   --gres=gpu:N    the widely supported GPU syntax; newer clusters also accept --gpus=N. For N GPUs on
#                   one node also set --ntasks-per-node=N (one process per GPU); gpus=N is passed below.
#   --mem=32G       generous for the reference configuration; --mem-per-gpu is not available everywhere.
#   --account       add `#SBATCH --account=<project>` where the site uses SLURM accounting. A bogus
#                   account makes every submission fail with `Invalid account`, so omit it otherwise.
#   -w <node>       pass on the command line to pin a run to a particular node.

set -euo pipefail

RUN_NAME="${1:?usage: sbatch scripts/train.sbatch <run name> [overrides...]}"
shift

# The experiment comes from the environment, not from "$@": hydra rejects two values for the same key,
# so a hard-coded experiment= plus a passthrough experiment= would be a composition error.
EXPERIMENT="${EDITFLOW_EXPERIMENT:-deletion-gpu-24h}"
VENV="${EDITFLOW_VENV:-$HOME/0envs/EditFlows_env}"
PYTHON="$VENV/bin/python"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

if [ ! -x "$PYTHON" ]; then
    echo "no interpreter at $PYTHON - build the environment or set EDITFLOW_VENV" >&2
    exit 1
fi

echo "job ${SLURM_JOB_ID:-?} on $(hostname), $(date)"
echo "experiment=$EXPERIMENT name=$RUN_NAME python=$PYTHON"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true

N_GPUS="${SLURM_GPUS_ON_NODE:-${SLURM_NTASKS_PER_NODE:-1}}"
N_WORKERS=$(( ${SLURM_CPUS_PER_TASK:-8} - 1 ))

srun "$PYTHON" train.py \
    experiment="$EXPERIMENT" \
    name="$RUN_NAME" \
    gpus="$N_GPUS" \
    data.num_workers="$N_WORKERS" \
    "$@"

echo "finished $(date)"
