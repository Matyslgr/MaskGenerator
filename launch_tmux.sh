#!/bin/bash

ID=$1
SCRIPT_PATH="${2:-scripts/launch.py}"

if [ -z "$ID" ]; then
  echo "Usage: $0 <session_id>"
  exit 1
fi

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
  echo "tmux is not installed. Please install it first."
  exit 1
fi

SESSION="train_${ID}"

CMD="
source ~/miniconda3/etc/profile.d/conda.sh && \
conda activate Robocar && \
PYTHONPATH=. python $SCRIPT_PATH
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