# Soft Computing Main Project

This repository contains the full course project pipeline for the **Epileptic Seizure Recognition** dataset.

## Colab (Recommended)

- Open directly in Colab: [epileptic_seizure_full_pipeline_colab.ipynb](https://colab.research.google.com/github/adhamhaithameid/soft-computing-main-project/blob/main/03_notebooks/colab/epileptic_seizure_full_pipeline_colab.ipynb)
- Notebook: `03_notebooks/colab/epileptic_seizure_full_pipeline_colab.ipynb`
- Run this notebook in Google Colab to execute:
  - data fetching
  - preprocessing
  - feature reduction and selection
  - model training and accuracy computation
  - paper draft generation

## Quick Start

1. Create/activate virtual environment:

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
python -m pip install --disable-pip-version-check -r requirements.txt
```

2. Run complete workflow:

```bash
./run_all.sh
```

3. Generated outputs (after dependencies are installed successfully):
- `05_results/metrics/metrics_all.csv`
- `05_results/tables/summary_accuracy.csv`
- `05_results/figures/*`
- `06_paper/draft/*`

## Notes

- `04_src/fetch_data.py` uses standard-library download/clean logic with retries and multi-source fallback.
- Current verified dataset snapshot: `11,500` rows.
- If you want to retry full dataset download:

```bash
python 04_src/fetch_data.py --force-download
```

## Structure

- `01_lectures/`: provided lectures used to design the project methods.
- `02_data/`: raw/interim/processed data and source links.
- `03_notebooks/`: Kaggle/local/Colab notebook artifacts.
- `04_src/`: reproducible Python pipeline.
- `05_results/`: metrics tables, fold outputs, and figures.
- `06_paper/`: template, draft sections, paper tables and figures.
- `07_docs/`: planning, method notes, and execution log.

## Top-Level Guides

- `PROJECT_MASTER_GUIDE.md`: full project explanation and progress.
- `FOLDER_STRUCTURE.md`: clean tree of folders/files.
- `ABOUT.md`: repository scope summary.
