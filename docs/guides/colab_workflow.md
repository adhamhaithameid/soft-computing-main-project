# Google Colab Workflow

## Notebook
- Path: `notebooks/colab/epileptic_seizure_full_pipeline_colab.ipynb`
- Direct URL: https://colab.research.google.com/github/adhamhaithameid/epileptic-seizure-recognition/blob/main/notebooks/colab/epileptic_seizure_full_pipeline_colab.ipynb

## How to run
1. Open the URL above.
2. Set runtime to Python 3.
3. Run all cells in order.
4. `RUN_ENV` is auto-detected in the environment cell (`colab` online, `local` on laptop). Override only if needed.
5. In the config cell, choose `RUN_PROFILE`:
   - `local_smoke` (quick)
   - `local_medium` (partial)
   - `full` (all 4608 fold evaluations)

## Run Colab UI + local machine compute
Use this when Colab cloud runtime is slow or disconnects during full Cartesian runs.

1. In local terminal at repo root:
   ```bash
   ./start_colab_local_runtime.sh
   ```
2. Copy printed URL (`http://localhost:8888/?token=...`).
3. In Colab top-right: `Connect` -> `Connect to local runtime`.
4. Paste URL and connect.
5. Run notebook cells from Colab page. Kernels execute locally.

Stop local runtime:
```bash
./stop_colab_local_runtime.sh
```

If you get a 404 from old links:
- old path: `soft-computing-main-project/03_notebooks/colab` (deleted)
- current path: `epileptic-seizure-recognition/notebooks/colab`

## What the notebook executes
- sets up environment (colab or local)
- installs dependencies (in colab mode)
- loads repo/workspace
- loads dataset and builds binary + multiclass tracks
- runs full Cartesian benchmark across all configured stages
- writes metrics/tables/figures/report
- optionally zips outputs into `colab_outputs.zip` (colab mode)

## Expected full-run counts
- `expected_combos = 1536`
- `expected_fold_evals = 4608`

## Main output files
- `results/metrics/cartesian_metrics_all.csv`
- `results/metrics/cartesian_run_manifest.json`
- `results/tables/cartesian_summary_by_combo.csv`
- `results/tables/cartesian_rankings_binary.csv`
- `results/tables/cartesian_rankings_multiclass.csv`
- `results/reports/cartesian_comparison_report.md`
- `results/figures/cartesian_*.png`
- `paper/draft/*.md`

## Skip/failure behavior
The run does not stop on invalid combinations. It logs them as:
- `status = failed`
- `skip_reason = <reason>`

This guarantees full coverage accounting, even when some combinations are mathematically invalid.
