# Current Status

## Implemented
- Refactored project directories to `assets/data/notebooks/src/results/paper/docs`.
- Added centralized path config (`src/config/paths.py`).
- Added staged Cartesian benchmark core with deterministic iteration.
- Added auto-fix/skip handling and failure reason logging.
- Added checkpoint/resume support for long Colab runs.
- Added comparison tables and report generation.
- Added graph suite (heatmaps, top-N bars, fold variance, ROC).
- Rebuilt Colab notebook for self-contained staged workflow.

## Current benchmark contract
- Tracks: binary + multiclass
- CV folds: 3
- Unique combinations: 1536
- Fold evaluations: 4608

## Main commands
```bash
python src/cli/fetch_data.py
python src/cli/check_env.py
python src/cli/run_experiments.py
python src/cli/generate_paper_drafts.py
```

## Main outputs
- `results/metrics/cartesian_metrics_all.csv`
- `results/metrics/cartesian_run_manifest.json`
- `results/tables/cartesian_summary_by_combo.csv`
- `results/tables/cartesian_rankings_binary.csv`
- `results/tables/cartesian_rankings_multiclass.csv`
- `results/reports/cartesian_comparison_report.md`
- `results/figures/cartesian_*.png`
