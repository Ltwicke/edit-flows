#!/bin/bash
# SLURM job template for training on a GPU node. Filled in for the UiO CCSE "Egil" cluster; see
# docs/CLUSTER.md for the site-neutral version of every value below.
#
#   mkdir -p runs                                        # ONCE: --output below is resolved at submit time
#   sbatch scripts/train.sbatch <run name> [hydra overrides...]
#
#   EDITFLOW_EXPERIMENT=deletion-3090-check sbatch scripts/train.sbatch gpu-check    # the plumbing check
#   sbatch scripts/train.sbatch long-01 trainer.max_epochs=8                         # ~30 minutes
#   sbatch --partition=<other> -w hugefacet --gres=gpu:4 --ntasks-per-node=4 \
#          scripts/train.sbatch big-01                                               # another node, 4 GPUs
#
# sbatch command-line options override the #SBATCH directives below, so the node, partition, GPU count
# and time limit are all changeable without editing this file.
#
# The run resumes automatically from its newest checkpoint when the same run name is submitted again, so
# a run that needs more than one job's time limit is simply resubmitted (see scripts/submit_chain.sh).

#SBATCH --job-name=editflow-deletion
#SBATCH --partition=normal
#SBATCH --nodelist=neuroai
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-23:30:00
#SBATCH --output=runs/slurm-%x-%j.out

# Notes on the directives above, for when this is moved to another site or node:
#   --nodelist      remove it to let SLURM choose, or override with `sbatch -w <node>`. If `scontrol show
#                   node neuroai` reports the node is unknown, neuroai is not managed by SLURM at all and
#                   the detached path in docs/CLUSTER.md applies instead.
#   --gres=gpu:N    this site's documented GPU syntax. Newer clusters also accept --gpus=N.
#   --mem=32G       generous for the reference configuration; --mem-per-gpu is not available everywhere.
#   --account       this site has no SLURM accounting, so there is no --account line. Add one where the
#                   site requires it.
# For N GPUs on one node: --gres=gpu:N --ntasks-per-node=N (one process per GPU); gpus=N is passed below.

set -euo pipefail

RUN_NAME="${1:?usage: sbatch scripts/train.sbatch <run name> [overrides...]}"
shift

# The experiment comes from the environment, not from "$@": hydra rejects two values for the same key, so
# a hard-coded experiment= plus a passthrough experiment= would be a composition error.
EXPERIMENT="${EDITFLOW_EXPERIMENT:-deletion-3090-24h}"
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
