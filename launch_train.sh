#!/bin/bash

set -e

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
  echo "tmux is not installed. Please install it first."
  exit 1
fi

# Arguments
RUN_ID=$1
SEARCH_MODE=$2  # "grid" ou "custom"

if [ -z "$RUN_ID" ]; then
  echo "Usage: $0 <run_id> [search_mode]"
  echo "Example: $0 001 grid"
  exit 1
fi

# Default search mode to 'grid' if not provided
if [ -z "$SEARCH_MODE" ]; then
  echo "No search mode provided. Defaulting to 'grid'."
  SEARCH_MODE="grid"
fi

# Constants
BRANCH="main"
SCRIPT_PATH="scripts/launch.py"
WORKTREES_DIR="$HOME/Worktrees"
BASE_DIR="$HOME/MaskGenerator"
WORKTREE="$WORKTREES_DIR/train_${RUN_ID}"
SESSION="train_${RUN_ID}"

mkdir -p "$WORKTREES_DIR"

echo "Launching training run $RUN_ID on $BRANCH branch"

cd "$BASE_DIR"
git fetch origin
git pull origin $BRANCH

if [ -d "$WORKTREE" ]; then
  echo "Worktree $WORKTREE already exists. Deleting."
  git worktree remove --force "$WORKTREE"
fi

git worktree add --detach "$WORKTREE" origin/$BRANCH

COMMIT_SHA=$(git -C "$WORKTREE" rev-parse HEAD)

echo "Using commit: $COMMIT_SHA"

CMD="
cd $WORKTREE && \
source /root/anaconda3/etc/profile.d/conda.sh && \
conda activate Robocar && \
PYTHONPATH=. python $SCRIPT_PATH --search $SEARCH_MODE; \
exec bash
"

# Check if the session already exists
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session $SESSION already exists. Attaching to it."
    tmux attach-session -t "$SESSION"
else
    echo "Creating new tmux session: $SESSION"
    tmux new-session -d -s "$SESSION" "$CMD"
    tmux attach-session -t "$SESSION"
fi
