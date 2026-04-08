#!/usr/bin/env python3
"""Run staged Cartesian benchmark for the soft-computing project."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core import CartesianSpec, RunnerIO, build_summary, save_comparisons  # noqa: E402
from src.core.benchmark import run_cartesian_benchmark  # noqa: E402
from src.config.paths import (  # noqa: E402
    DATA_PROCESSED_DIR,
    DATASET_CSV,
    RESULTS_FIGURES_DIR,
    RESULTS_METRICS_DIR,
    RESULTS_REPORTS_DIR,
    RESULTS_TABLES_DIR,
)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    HAS_PLOTTING = True
except Exception:
    HAS_PLOTTING = False


def load_dataset() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if not DATASET_CSV.exists():
        raise FileNotFoundError(
            f"Raw CSV not found at {DATASET_CSV}. Run: python src/cli/fetch_data.py"
        )

    df = pd.read_csv(DATASET_CSV)

    unnamed = [c for c in df.columns if c.lower().startswith("unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

    target_col = "y" if "y" in df.columns else df.columns[-1]

    for col in df.columns:
        if col != target_col:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    X = df.drop(columns=[target_col]).copy()
    X = X.fillna(X.median(numeric_only=True))

    y_multi = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int).to_numpy()
    y_binary = (y_multi == 1).astype(int)

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    X.to_csv(DATA_PROCESSED_DIR / "features_numeric.csv", index=False)
    pd.DataFrame({"y_multiclass": y_multi, "y_binary": y_binary}).to_csv(
        DATA_PROCESSED_DIR / "targets.csv", index=False
    )

    return X, y_multi, y_binary


def save_statistical_analysis(X: pd.DataFrame) -> None:
    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    desc = pd.DataFrame(
        {
            "min": X.min(),
            "max": X.max(),
            "mean": X.mean(),
            "variance": X.var(),
            "std": X.std(),
            "skewness": X.apply(lambda s: skew(s.to_numpy(), bias=False)),
            "kurtosis": X.apply(lambda s: kurtosis(s.to_numpy(), bias=False)),
        }
    )
    desc.to_csv(RESULTS_TABLES_DIR / "dataset_descriptive_stats.csv")

    cov = X.cov()
    corr = X.corr()
    cov.to_csv(RESULTS_TABLES_DIR / "covariance_matrix.csv")
    corr.to_csv(RESULTS_TABLES_DIR / "correlation_matrix.csv")

    if HAS_PLOTTING:
        plt.figure(figsize=(14, 10))
        sns.heatmap(corr, cmap="coolwarm", center=0)
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(RESULTS_FIGURES_DIR / "correlation_heatmap.png", dpi=200)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap for smoke testing.")
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--fresh", action="store_true", help="Start fresh and overwrite old metrics file.")
    args = parser.parse_args()

    X_df, y_multi, y_binary = load_dataset()
    save_statistical_analysis(X_df)

    RESULTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_csv = RESULTS_METRICS_DIR / "cartesian_metrics_all.csv"
    manifest_json = RESULTS_METRICS_DIR / "cartesian_run_manifest.json"

    if args.fresh:
        metrics_csv.unlink(missing_ok=True)
        manifest_json.unlink(missing_ok=True)

    spec = CartesianSpec(cv_splits=3)
    io = RunnerIO(
        metrics_csv=metrics_csv,
        manifest_json=manifest_json,
        checkpoint_every=max(1, args.checkpoint_every),
    )

    df = run_cartesian_benchmark(
        X_df=X_df,
        y_binary=y_binary,
        y_multiclass=y_multi,
        spec=spec,
        io=io,
        resume=not args.fresh,
        max_rows=args.max_rows,
    )

    ok = df[df["status"] == "ok"].copy()
    summary = build_summary(ok).sort_values(["track", "accuracy"], ascending=[True, False])
    summary.to_csv(RESULTS_TABLES_DIR / "cartesian_summary_by_combo.csv", index=False)
    saved = save_comparisons(summary, RESULTS_TABLES_DIR, RESULTS_REPORTS_DIR)

    print("Cartesian benchmark completed.")
    print(f"Expected combos: {spec.expected_combos}")
    print(f"Expected fold evals: {spec.expected_fold_evals}")
    print(f"Saved metrics: {metrics_csv}")
    print(f"Saved manifest: {manifest_json}")
    print(f"Saved summary: {RESULTS_TABLES_DIR / 'cartesian_summary_by_combo.csv'}")
    print(f"Saved binary rankings: {saved['binary']}")
    print(f"Saved multiclass rankings: {saved['multiclass']}")
    print(f"Saved comparison report: {saved['report']}")


if __name__ == "__main__":
    main()
