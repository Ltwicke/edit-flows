#!/bin/bash
# SLURM job template for training on GPU nodes. Fill in the placeholders in <...> for your site.
#
#   sbatch scripts/train.sbatch <run name> [extra hydra overrides...]
#
# The run resumes automatically from its newest checkpoint when the same job is submitted again, so a run
# that needs more than one job's time limit is simply resubmitted (see scripts/submit_chain.sh).

#SBATCH --job-name=editflow-deletion
#SBATCH --account=<project-account>
#SBATCH --partition=<gpu-partition>
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=32G
#SBATCH --time=0-23:30:00
#SBATCH --output=runs/slurm-%x-%j.out

# For N GPUs on one node: --gpus=N --ntasks-per-node=N (one process per GPU) and gpus=N below.
# To pick a GPU type use the site's syntax, e.g. --gpus=<type>:N.

set -euo pipefail

RUN_NAME="${1:?usage: sbatch scripts/train.sbatch <run name> [overrides...]}"
shift

module purge
# module load <Python 3.13 module of your site>
source "<path to>/EditFlows_env/bin/activate"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

echo "job ${SLURM_JOB_ID:-?} on $(hostname), $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

N_GPUS="${SLURM_GPUS_ON_NODE:-${SLURM_NTASKS_PER_NODE:-1}}"
N_WORKERS=$(( ${SLURM_CPUS_PER_TASK:-8} - 1 ))

srun python train.py \
    experiment=deletion-base \
    name="$RUN_NAME" \
    gpus="$N_GPUS" \
    data.num_workers="$N_WORKERS" \
    "$@"

echo "finished $(date)"
