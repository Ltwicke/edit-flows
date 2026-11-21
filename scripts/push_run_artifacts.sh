#!/bin/bash
# Push the small text artifacts of one run to their own branch, so they can be analysed on a machine that
# cannot reach the cluster with scp or rsync. Checkpoints stay on the cluster; use evaluate.py there.
#
#   scripts/push_run_artifacts.sh <run name> [remote]
#
# On the analysis machine, drop the files into the working tree without changing branch (they stay
# ignored locally, because /runs/ is in .gitignore):
#
#   git fetch origin
#   git restore --source origin/run-artifacts/<run name> -- runs/
#
# This never switches branch, never touches the index and never writes to the working tree, so it is safe
# to run while a training job is in flight. That matters: checking the artifact branch out and back again
# would DELETE the live run directory, because these files are tracked there and ignored here.
set -euo pipefail

RUN_NAME="${1:?usage: scripts/push_run_artifacts.sh <run name> [remote]}"
REMOTE="${2:-origin}"

cd "$(dirname "$0")/.."

RUN_DIR="runs/$RUN_NAME"
BRANCH="run-artifacts/$RUN_NAME"

[ -d "$RUN_DIR" ] || { echo "no such run directory: $RUN_DIR" >&2; exit 1; }

# The text artifacts only: resolved config, the appended log of every (re)start, lightning's per-step CSV
# of every restart, and the per-epoch deletion metrics. Deliberately not checkpoints/ (large) and not
# eval/val_epoch_NNNN.json (one file per epoch; deletion_metrics.csv already holds every row).
mapfile -t FILES < <(
    find "$RUN_DIR" -maxdepth 1 -type f \( -name config.yaml -o -name train.log \)
    find "$RUN_DIR/logs" -type f \( -name metrics.csv -o -name hparams.yaml \) 2>/dev/null
    find "$RUN_DIR/eval" -maxdepth 1 -type f -name deletion_metrics.csv 2>/dev/null
)

[ "${#FILES[@]}" -gt 0 ] || { echo "no artifacts found under $RUN_DIR" >&2; exit 1; }

echo "pushing ${#FILES[@]} files ($(du -ch "${FILES[@]}" | tail -1 | cut -f1)) to $REMOTE $BRANCH"

# Build the commit with a throwaway index so the real index and HEAD are untouched.
GIT_INDEX_FILE="$(git rev-parse --git-dir)/push-artifacts.index"
export GIT_INDEX_FILE
trap 'rm -f "$GIT_INDEX_FILE"' EXIT
rm -f "$GIT_INDEX_FILE"
git read-tree --empty
git add -f -- "${FILES[@]}"
TREE="$(git write-tree)"

# Accumulate history on the branch when it already exists, so repeated pushes fast-forward.
git fetch -q "$REMOTE" "$BRANCH" 2>/dev/null || true
PARENT=()
if git rev-parse --verify --quiet FETCH_HEAD >/dev/null 2>&1; then
    PARENT=(-p "$(git rev-parse FETCH_HEAD)")
fi

if ! git var GIT_AUTHOR_IDENT >/dev/null 2>&1; then
    GIT_AUTHOR_NAME="editflow-cluster"; GIT_AUTHOR_EMAIL="editflow@$(hostname)"
    GIT_COMMITTER_NAME="$GIT_AUTHOR_NAME"; GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"
    export GIT_AUTHOR_NAME GIT_AUTHOR_EMAIL GIT_COMMITTER_NAME GIT_COMMITTER_EMAIL
fi

EPOCHS="unknown"
if [ -f "$RUN_DIR/eval/deletion_metrics.csv" ]; then
    EPOCHS="$(awk -F, 'NR>1 && $4=="all" {print $2}' "$RUN_DIR/eval/deletion_metrics.csv" | sort -n | tail -1)"
fi

COMMIT="$(git commit-tree "$TREE" "${PARENT[@]}" -m "results($RUN_NAME): through epoch $EPOCHS on $(hostname)")"
git push "$REMOTE" "$COMMIT:refs/heads/$BRANCH"

echo "pushed $BRANCH ($COMMIT)"
