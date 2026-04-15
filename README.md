# Soft Computing Course Project

End-to-end benchmark lab for **Epileptic Seizure Recognition** with:
- full Cartesian benchmarking (1536 combinations, 4608 fold evaluations),
- CPU or GPU execution modes,
- automated checkpoints every 5% progress,
- reproducible outputs (metrics, tables, figures, reports),
- paper drafting assets mapped to the course template.

## Scope Lock
This repository is dedicated to **one use case only**: Epileptic Seizure Recognition.
All scripts, outputs, and documentation are optimized for this dataset and task.

## 1) What This Project Delivers
- Preprocessing comparison: `standard`, `minmax`, `robust`, `quantile`
- Feature reduction: `none`, `pca`, `lda_projection`, `svd`
- Feature selection:
  - Filter: `filter_chi2`, `filter_anova`, `filter_correlation`
  - Wrapper: `wrapper_sfs`, `wrapper_rfe`
  - Embedded: `embedded_l1`
  - Evolutionary: `ga_selection`
- Classifiers: `knn`, `svm`, `decision_tree`, `logistic_regression`, `lda_classifier`, `mlp_ann`
- Tracks: `binary` + `multiclass`
- CV: `3` folds

Combination count:
- Unique combinations: `4 x 4 x 8 x 6 x 2 = 1536`
- Fold evaluations: `1536 x 3 = 4608`

## 2) Quick Start
```bash
python -m venv .venv311
source .venv311/bin/activate   # Windows PowerShell: .venv311\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Run interactive launcher:
```bash
python run_all.py
```

You will choose:
1. execution mode (`cpu` or `gpu`)
2. platform profile (`linux`, `windows`, or `mac`)

## 3) Non-Interactive Examples
Full CPU run:
```bash
python run_all.py --mode cpu --platform-profile linux --non-interactive --fresh
```

Full GPU run (fallback allowed):
```bash
python run_all.py --mode gpu --platform-profile linux --non-interactive --fresh
```

Strict GPU run (fails if GPU acceleration is unavailable):
```bash
python run_all.py --mode gpu --strict-device --platform-profile linux --non-interactive --fresh
```

Smoke test:
```bash
python run_all.py --mode cpu --platform-profile mac --non-interactive --fresh --max-rows 300 --allow-partial
```

## 4) Checkpoints and Progress
The benchmark writes durable checkpoints and prints terminal progress at every **5%** by default.

You can change this:
```bash
python run_all.py --checkpoint-percent 10 --non-interactive --mode cpu --platform-profile linux
```

## 5) Run History (Automatic)
Each `run_all.py` execution creates:
- a run archive folder: `results/history/runs/runN_<timestamp>/`
- a global run index: `results/history/RUN_HISTORY.md`
- a machine-readable history file: `results/history/run_history.json`

Each run archive stores:
- manifest snapshot,
- validation report,
- comparison report,
- run summary with timing and configuration.

## 6) Output Contracts
Main metrics file:
- `results/metrics/cartesian_metrics_all.csv`
  - `track, fold, preprocessing, reduction, selection, model`
  - `accuracy, precision, recall, f1, roc_auc, error_rate`
  - `fit_time_sec, predict_time_sec, status, skip_reason`

Manifest file:
- `results/metrics/cartesian_run_manifest.json`
  - `expected_combos, expected_fold_evals, target_total_rows`
  - `rows_written, completed_ok, skipped_or_failed`
  - `runtime_sec, started_utc, finished_utc`
  - `checkpoint_percent, run_label, platform_profile`
  - `execution_device, acceleration_backend`
  - `best_binary, best_multiclass`

Validation report:
- `results/reports/cartesian_validation_report.md`

## 7) Main Entrypoints
- `run_all.py` (recommended launcher)
- `src/cli/fetch_data.py`
- `src/cli/check_env.py`
- `src/cli/run_experiments.py`
- `src/cli/validate_cartesian_outputs.py`
- `src/cli/generate_paper_drafts.py`

## 8) Colab Notebook
- `notebooks/colab/epileptic_seizure_full_pipeline_colab.ipynb`
- Notebook supports both environments:
  - `RUN_ENV = "colab"` for Google Colab
  - `RUN_ENV = "local"` for local Jupyter execution

### Run Colab UI with local compute (recommended for long runs)
This lets you monitor in the Colab browser page while computation runs on your laptop.

1. Start local runtime from this repo:
```bash
./start_colab_local_runtime.sh
```

2. Copy the printed URL (for example `http://localhost:8888/?token=...`).
3. Open the notebook online:
   - https://colab.research.google.com/github/adhamhaithameid/epileptic-seizure-recognition/blob/main/notebooks/colab/epileptic_seizure_full_pipeline_colab.ipynb
4. In Colab: `Connect` -> `Connect to local runtime` -> paste the URL.

Stop runtime when done:
```bash
./stop_colab_local_runtime.sh
```

Notes:
- The old repo path `soft-computing-main-project/.../03_notebooks/colab` returns `404` because this project now uses:
  `epileptic-seizure-recognition/notebooks/colab/...`
- Local runtime logs are written to `.colab_runtime.log`.

## 9) Paper Assets
- Template: `paper/template/`
- Drafts: `paper/draft/`
- Final mapped draft: `paper/draft/09_full_paper_draft_mapped_to_template.md`
- Full draft copy: `RESEARCH_PAPER_FINAL_DRAFT.md`

## 10) Additional Guides
- `PROJECT_MASTER_GUIDE.md`
- `FOLDER_STRUCTURE.md`
- `ABOUT.md`
