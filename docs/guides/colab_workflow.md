# Google Colab Workflow

## Notebook
- Path: `notebooks/colab/epileptic_seizure_full_pipeline_colab.ipynb`
- Direct URL: https://colab.research.google.com/github/adhamhaithameid/epileptic-seizure-recognition-soft-computing/blob/main/notebooks/colab/epileptic_seizure_full_pipeline_colab.ipynb

## How to run
1. Open the URL above.
2. Set runtime to Python 3.
3. Run all cells in order.
4. Choose full run (`SMOKE_TEST=False`) or smoke run (`SMOKE_TEST=True`) in the config cell.

## What the notebook executes
- installs dependencies
- loads repo/workspace
- loads dataset and builds binary + multiclass tracks
- runs full Cartesian benchmark across all configured stages
- writes metrics/tables/figures/report
- zips outputs into `colab_outputs.zip`

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
