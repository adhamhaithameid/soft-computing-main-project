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

## Latest Verified Full Run (Apple Silicon M1, April 9, 2026)
- Runtime: `5294.26 sec` (`88.24 min`)
- Evaluation accounting:
  - `expected_fold_evals = 4608`
  - `completed_ok = 4392`
  - `skipped_or_failed = 216`
- Execution mode:
  - `execution_device = cpu`
  - `acceleration_backend = none`
- Best binary pipeline:
  - `svm + quantile + pca + none`
  - `accuracy = 0.976261`, `f1 = 0.939349`, `roc_auc = 0.995438`
- Best multiclass pipeline:
  - `mlp_ann + minmax + pca + none`
  - `accuracy = 0.685651`, `f1 = 0.685026`

Top skip reasons (`216` total):
- `72`: `selection_failed: ValueError: Found array with 1 feature(s) ... minimum of 2 is required by SequentialFeatureSelector.`
- `72`: `selection_failed: ValueError: n_features_to_select must be < n_features.`
- `48`: `selection_failed: ValueError: Found array with 1 feature(s) ... minimum of 2 is required by RFE. (shape=(7667, 1))`
- `24`: `selection_failed: ValueError: Found array with 1 feature(s) ... minimum of 2 is required by RFE. (shape=(7666, 1))`

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
