# Soft Computing Main Project - Master Guide

## Repository
- GitHub: https://github.com/adhamhaithameid/soft-computing-main-project
- Colab notebook: https://colab.research.google.com/github/adhamhaithameid/soft-computing-main-project/blob/main/notebooks/colab/epileptic_seizure_full_pipeline_colab.ipynb

## 1) Goal
Build a complete Soft Computing course project with:
- dataset decision
- preprocessing
- feature reduction
- feature selection
- classifier benchmarking
- accuracy/F1 comparisons and plots
- paper-ready draft outputs

## 2) Dataset Decision
Selected dataset: **Epileptic Seizure Recognition**.

Reason:
- Tabular structure is easiest to preprocess.
- Works cleanly with PCA/LDA/SVD and all requested selection methods.
- Supports both binary and multiclass tracks.

## 3) Refactored Architecture
- Lectures: `assets/lectures/`
- Data: `data/raw`, `data/interim`, `data/processed`, `data/catalog`
- Notebooks: `notebooks/colab`, `notebooks/kaggle`, `notebooks/local`
- Source code: `src/config`, `src/core`, `src/cli`
- Outputs: `results/metrics`, `results/tables`, `results/figures`, `results/folds`, `results/reports`
- Paper files: `paper/template`, `paper/draft`, `paper/tables`, `paper/figures`, `paper/references`
- Documentation: `docs/plans`, `docs/guides`, `docs/status`, `docs/paper`

## 4) Cartesian Benchmark Contract
- Tracks: `binary`, `multiclass`
- CV folds: `3`
- Preprocessing: `4` methods
- Reduction: `4` methods
- Selection: `8` methods
- Classifiers: `6` methods

Combination count:
- Unique combos: `4 x 4 x 8 x 6 x 2 = 1536`
- Fold evaluations: `1536 x 3 = 4608`

## 5) Core Outputs
- `results/metrics/cartesian_metrics_all.csv`
- `results/metrics/cartesian_run_manifest.json`
- `results/tables/cartesian_summary_by_combo.csv`
- `results/tables/cartesian_rankings_binary.csv`
- `results/tables/cartesian_rankings_multiclass.csv`
- `results/reports/cartesian_comparison_report.md`
- `results/figures/cartesian_*.png`

## 6) How to Run
```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
python -m pip install --disable-pip-version-check -r requirements.txt
./run_all.sh
```

Or run the Colab notebook in `notebooks/colab/`.

## 7) Where To Read Next
- Folder map: `FOLDER_STRUCTURE.md`
- Colab workflow: `docs/guides/colab_workflow.md`
- Lecture understanding: `docs/guides/lecture_understanding.md`
- Current status: `docs/status/current_status.md`
- Migration notes: `docs/status/migration_notes.md`
- Paper writing plan: `docs/paper/paper_writing_plan.md`
