# Google Colab Workflow

This is the main cloud execution path for the project.

## Notebook
- Colab notebook path: `03_notebooks/colab/epileptic_seizure_full_pipeline_colab.ipynb`
- Direct URL: https://colab.research.google.com/github/adhamhaithameid/soft-computing-main-project/blob/main/03_notebooks/colab/epileptic_seizure_full_pipeline_colab.ipynb

## How to run in Colab
1. Open Google Colab.
2. Open the direct URL above (recommended).
3. If needed, open and run:
   - `03_notebooks/colab/epileptic_seizure_full_pipeline_colab.ipynb`
4. Execute cells in order.

## What the notebook does
- Installs required dependencies.
- Resolves project directory automatically.
- Runs environment check.
- Downloads/refreshes dataset.
- Runs full experiment pipeline:
  - preprocessing
  - feature reduction
  - feature selection
  - classification models
  - evaluation metrics
- Generates paper draft markdown sections.
- Creates `colab_outputs.zip` for download.

## Expected outputs
- `05_results/metrics/metrics_all.csv`
- `05_results/metrics/run_summary.json`
- `05_results/folds/fold_metrics.csv`
- `05_results/tables/summary_accuracy.csv`
- `05_results/tables/confusion_matrix_best_binary.csv`
- `05_results/tables/confusion_matrix_best_multiclass.csv`
- `06_paper/draft/*.md`

## Notes
- The pipeline currently uses a lightweight runtime configuration (3-fold CV and reduced wrapper/GA search size) so it completes quickly in Colab and local runs.
- If plotting packages are installed (`matplotlib`, `seaborn`), figure files are generated too.
- Dataset fetch uses multiple source links with retry logic and integrity checks.
