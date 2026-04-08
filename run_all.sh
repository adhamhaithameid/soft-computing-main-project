#!/usr/bin/env zsh
set -e

if [[ -f ".venv311/bin/activate" ]]; then
  source .venv311/bin/activate
elif [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
fi

python src/cli/fetch_data.py
python src/cli/check_env.py
python src/cli/run_experiments.py
python src/cli/generate_paper_drafts.py

echo "All steps completed."
