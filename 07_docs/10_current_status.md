# Current Status

## Implemented
- Full project structure created.
- Lectures and paper template organized.
- Planning markdown files created.
- Data fetch/clean script created (`04_src/fetch_data.py`).
- Experiment pipeline created (`04_src/run_experiments.py`).
- Paper draft generator created (`04_src/generate_paper_drafts.py`).
- Kaggle/local notebook templates created.
- Google Colab notebook created (`03_notebooks/colab/epileptic_seizure_full_pipeline_colab.ipynb`).
- Lecture understanding summary added (`07_docs/11_lecture_understanding.md`).
- Single-command runner created (`run_all.sh`).
- Full local experiment run completed and outputs generated.
- GitHub repository created and pushed:
  - `https://github.com/adhamhaithameid/soft-computing-main-project`

## Data Snapshot
- Local CSV is now the full cleaned dataset with `11,500` rows and `180` columns (including target).
- Current run processed `11,500` rows with `178` features after dropping non-feature columns.
- Full dataset can be re-fetched using:

```bash
python 04_src/fetch_data.py --force-download
```

## Execution Commands

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
python -m pip install --disable-pip-version-check -r requirements.txt
./run_all.sh
```

## Latest Output Snapshot (2026-04-07)
- `05_results/metrics/metrics_all.csv` generated.
- `05_results/tables/summary_accuracy.csv` generated.
- `05_results/folds/fold_metrics.csv` generated.
- Best binary setup: `SVM + svd` with mean accuracy `0.9719`.
- Best multiclass setup: `MLP_ANN + original` with mean accuracy `0.6747`.
- Paper draft sections generated in `06_paper/draft/`.
