# Migration Notes (Legacy -> Refactor)

## Path mapping
- `01_lectures/*` -> `assets/lectures/*`
- `02_data/*` -> `data/*`
- `03_notebooks/*` -> `notebooks/*`
- `04_src/*` -> `src/cli/*` and `src/core/*`
- `05_results/*` -> `results/*`
- `06_paper/*` -> `paper/*`
- `07_docs/*` -> `docs/{plans,guides,status,paper}/*`

## Output mapping
- `results/metrics/metrics_all.csv` -> `results/metrics/cartesian_metrics_all.csv`
- `results/metrics/run_summary.json` -> `results/metrics/cartesian_run_manifest.json`
- `results/tables/summary_accuracy.csv` -> `results/tables/cartesian_summary_by_combo.csv`
- new ranking outputs:
  - `results/tables/cartesian_rankings_binary.csv`
  - `results/tables/cartesian_rankings_multiclass.csv`
- new report output:
  - `results/reports/cartesian_comparison_report.md`

## Validation command
```bash
python src/cli/validate_cartesian_outputs.py
```

Use `--allow-partial` for smoke runs with `--max-rows`.
