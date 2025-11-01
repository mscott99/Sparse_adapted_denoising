#!/bin/bash

REMOTE_REPO_PATH="/home/matthew/Prog/Sparse_adapted_denoising"  # ← UPDATE: Full path to repo on server
REMOTE_SCRIPT="run_server.py"       # ← Relative to repo root
LOCAL_PICKLE="/Users/matthewscott/Prog/Sparse_adapted_denoising/data/last_server.pkl"
BRANCH="${1:-sparse_again}"                    # ← Branch arg (default: main)

echo "🚀 Pulling $BRANCH & fetching DF"

ssh ub "
  cd '$REMOTE_REPO_PATH' &&
  source /home/matthew/.sparse_env > /dev/null&&
  source ./.venv/bin/activate > /dev/null &&
  git checkout '$BRANCH'>/dev/null &&
  git pull origin '$BRANCH'>/dev/null &&
  python3 '$REMOTE_SCRIPT'
" > "$LOCAL_PICKLE"

if [[ $? -eq 0 && -s "$LOCAL_PICKLE" ]]; then
  echo "✅ Success! ($BRANCH) → $LOCAL_PICKLE ($(ls -lh "$LOCAL_PICKLE" | awk '{print $5}'))"
  echo "Load: pd.read_pickle('$LOCAL_PICKLE')"
else
  echo "❌ Failed (check git/SSH)" >&2
  rm -f "$LOCAL_PICKLE"
  exit 1
fi
