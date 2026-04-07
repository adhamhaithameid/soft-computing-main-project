# Kaggle and Colab Links and Runs

- Dataset link: https://www.kaggle.com/datasets/harunshimanto/epileptic-seizure-recognition
- Notebook path: `03_notebooks/kaggle/epileptic_full_pipeline.ipynb`
- Colab notebook path: `03_notebooks/colab/epileptic_seizure_full_pipeline_colab.ipynb`

Run log entries should be appended here with:
- Date/time
- Notebook platform/version
- Main results snapshot

## Run Log

- 2026-04-07: Local scaffold created. Data fetch script executed. Full experiment execution pending dependency installation.
- 2026-04-07: Local full pipeline run completed successfully (3-fold CV lightweight config).
  - Binary best: `SVM + SVD`, accuracy `0.9397`, AUC `0.9836`.
  - Multiclass best: `SVM + Original`, accuracy `0.4184`.
  - Paper draft sections generated in `06_paper/draft/`.
- 2026-04-07: Full dataset re-downloaded (`11,500` rows) and pipeline re-run.
  - Binary best: `SVM + SVD`, accuracy `0.9719`, AUC `0.9945`.
  - Multiclass best: `MLP_ANN + Original`, accuracy `0.6747`.
  - Updated paper draft sections generated in `06_paper/draft/`.
- 2026-04-07: Repository published on GitHub.
  - Repo: `https://github.com/adhamhaithameid/soft-computing-main-project`
  - Colab direct notebook URL configured and documented.
