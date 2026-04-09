# Project Master Guide

## 1) Project Purpose
This repository is the Soft Computing course main project.  
It benchmarks a full pipeline for **Epileptic Seizure Recognition** and produces:
- metrics,
- comparisons,
- figures,
- validation reports,
- paper-ready draft content.

## Scope Lock
The project is intentionally constrained to **Epileptic Seizure Recognition only**.
No multi-dataset switching is part of the production workflow.

## 2) Dataset Decision
Selected dataset: **Epileptic Seizure Recognition**.

Why this dataset:
- tabular numeric structure is easy to preprocess,
- supports all requested reduction/selection methods,
- supports both binary and multiclass tasks.

## 3) Benchmark Design
- Tracks: `binary`, `multiclass`
- CV folds: `3`
- Preprocessing: `4`
- Reduction: `4`
- Selection: `8`
- Classifiers: `6`

Combinations:
- Unique combos: `1536`
- Fold evaluations: `4608`

## 4) Run Modes
You can run in:
1. `cpu` mode
2. `gpu` mode

Use interactive launcher:
```bash
python run_all.py
```

Or explicit command:
```bash
python run_all.py --mode cpu --platform-profile linux --non-interactive --fresh
```

## 5) Progress + Checkpoints
- Terminal progress and durable checkpoints are written every **5%** by default.
- Configurable via `--checkpoint-percent`.

## 6) Validation Report
Validation is generated automatically in:
- `results/reports/cartesian_validation_report.md`

The report includes:
- expected vs actual counts,
- pass/fail state,
- runtime,
- device/backend,
- skip-reason summary.

## 7) Run History
Each launcher execution is archived as:
- `results/history/runs/runN_<timestamp>/`

Global run index:
- `results/history/RUN_HISTORY.md`
- `results/history/run_history.json`

## 8) Main Outputs
- `results/metrics/cartesian_metrics_all.csv`
- `results/metrics/cartesian_run_manifest.json`
- `results/tables/cartesian_summary_by_combo.csv`
- `results/tables/cartesian_rankings_binary.csv`
- `results/tables/cartesian_rankings_multiclass.csv`
- `results/reports/cartesian_comparison_report.md`
- `results/reports/cartesian_validation_report.md`
- `results/figures/cartesian_*.png`

## 9) Paper Deliverables
- `paper/draft/09_full_paper_draft_mapped_to_template.md`
- `RESEARCH_PAPER_FINAL_DRAFT.md`

## 10) Suggested Workflow for Course Submission
1. Run full benchmark (`run_all.py`).
2. Confirm validation PASS.
3. Inspect top tables/figures.
4. Update paper intro/discussion wording if needed.
5. Export selected tables/figures into your final submission package.

## 11) External Best-Practice Mapping
- See `docs/guides/similar_projects_playbook.md` for a practical map of what strong public seizure-AI projects usually implement and what to adopt next here.
