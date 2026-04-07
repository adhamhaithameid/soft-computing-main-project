# Soft Computing Main Project - Master Guide

This file is the complete guide for the project.

## Repository
- GitHub: https://github.com/adhamhaithameid/soft-computing-main-project
- Direct Colab notebook: https://colab.research.google.com/github/adhamhaithameid/soft-computing-main-project/blob/main/03_notebooks/colab/epileptic_seizure_full_pipeline_colab.ipynb

## 1) Project Goal
Build a full Soft Computing course project using one dataset, including:
- preprocessing
- feature reduction
- feature selection
- model training
- accuracy/metrics computation
- results tables/graphs
- research-paper draft writing

## 2) Final Dataset Choice
Chosen dataset: **Epileptic Seizure Recognition**.

Why this dataset was selected:
- Tabular data, so preprocessing is straightforward.
- Works cleanly with PCA/LDA/SVD, filter/wrapper/embedded/GA feature selection, and common classifiers.
- Easy to run in local Python and Google Colab.

## 3) What Has Been Implemented
- Full folder structure for lectures, data, notebooks, source, results, and paper.
- Lecture files organized in `01_lectures/`.
- Data links and source list prepared in `02_data/links.md`.
- Robust dataset fetch script with retry + mirror fallback: `04_src/fetch_data.py`.
- End-to-end experiment pipeline: `04_src/run_experiments.py`.
- Paper draft generator from results: `04_src/generate_paper_drafts.py`.
- Google Colab notebook for full execution: `03_notebooks/colab/epileptic_seizure_full_pipeline_colab.ipynb`.
- Lecture-understanding summary: `07_docs/11_lecture_understanding.md`.
- Current-status and run logs updated in `07_docs/`.

## 4) Verified Data Status
- Raw CSV path: `02_data/raw/epileptic_seizure_recognition/epileptic_seizure_data.csv`
- Verified rows: **11,500**
- Verified columns: **180** (including target)
- Features used by pipeline: **178**

## 5) Verified Latest Results (Full Dataset)
From `05_results/metrics/run_summary.json`:
- Best binary model: `SVM + SVD`
  - Accuracy: **0.9719**
  - ROC-AUC: **0.9945**
- Best multiclass model: `MLP_ANN + Original Features`
  - Accuracy: **0.6747**

## 6) How to Run

### Local
```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
python -m pip install -r requirements.txt
./run_all.sh
```

### Google Colab (recommended)
Open and run:
- `03_notebooks/colab/epileptic_seizure_full_pipeline_colab.ipynb`

Detailed Colab steps:
- `07_docs/12_colab_workflow.md`

## 7) Main Outputs
- `05_results/metrics/metrics_all.csv`
- `05_results/metrics/run_summary.json`
- `05_results/folds/fold_metrics.csv`
- `05_results/tables/summary_accuracy.csv`
- `05_results/tables/confusion_matrix_best_binary.csv`
- `05_results/tables/confusion_matrix_best_multiclass.csv`
- `06_paper/draft/*.md`

## 8) Folder Structure
See:
- `FOLDER_STRUCTURE.md`

## 9) Notes
- The UCI direct ZIP endpoint currently fails (HTTP 404 in this environment), so the fetch script uses working mirrors automatically.
- Current runtime config is optimized to finish reliably (3-fold CV + lighter wrapper/GA settings).
