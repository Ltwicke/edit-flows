#!/bin/bash
# Submit N dependent copies of scripts/train.sbatch, each starting after the previous one ended (for any
# reason). Since a run resumes from its newest checkpoint, this continues training across time limits.
#
#   scripts/submit_chain.sh <number of jobs> <run name> [extra hydra overrides...]

set -euo pipefail

N_JOBS="${1:?usage: scripts/submit_chain.sh <number of jobs> <run name> [overrides...]}"
RUN_NAME="${2:?usage: scripts/submit_chain.sh <number of jobs> <run name> [overrides...]}"
shift 2

cd "$(dirname "$0")/.."

previous=""
for i in $(seq 1 "$N_JOBS"); do
    if [ -z "$previous" ]; then
        job_id=$(sbatch --parsable scripts/train.sbatch "$RUN_NAME" "$@")
    else
        job_id=$(sbatch --parsable --dependency=afterany:"$previous" scripts/train.sbatch "$RUN_NAME" "$@")
    fi
    echo "job $i/$N_JOBS: $job_id${previous:+ (after $previous)}"
    previous="$job_id"
done
