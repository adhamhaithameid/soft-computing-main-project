# Colab Notebook

- Main notebook: `epileptic_seizure_full_pipeline_colab.ipynb`
- Direct Colab URL: https://colab.research.google.com/github/adhamhaithameid/epileptic-seizure-recognition/blob/main/notebooks/colab/epileptic_seizure_full_pipeline_colab.ipynb

## Purpose
Run the staged Cartesian benchmark with a step-by-step notebook that supports both Google Colab and local Jupyter execution:
- Preprocessing: Standard, MinMax, Robust, Quantile
- Reduction: none, PCA, LDA projection, SVD
- Selection: none, chi2, ANOVA, correlation, SFS, RFE, embedded L1, GA
- Classifiers: KNN, SVM, Decision Tree, Logistic Regression, LDA classifier, MLP/ANN

## Run Modes in Notebook
- `RUN_ENV` is auto-detected:
  - online Colab -> `colab`
  - local Jupyter/VS Code -> `local`
- You can still override manually if needed.
- Use `RUN_PROFILE`:
  - `local_smoke` for fast checks
  - `local_medium` for practical partial runs
  - `full` for the complete benchmark

## Outputs
The notebook writes outputs under `results/*` and `paper/draft/*`.
In Colab mode it also packages artifacts as `colab_outputs.zip`.

## Local Laptop Runbook
If you want to run the same benchmark on your own laptop (Arch Linux, macOS, or Windows), use the exact commands in:
- root `README.md` -> section **Run on Any Laptop (Exact Steps)**

See detailed Colab steps in `docs/guides/colab_workflow.md`.
