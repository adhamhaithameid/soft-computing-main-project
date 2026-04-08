from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


def build_summary(ok_df: pd.DataFrame) -> pd.DataFrame:
    return (
        ok_df.groupby(["track", "preprocessing", "reduction", "selection", "model"], as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            f1=("f1", "mean"),
            roc_auc=("roc_auc", "mean"),
            fit_time_sec=("fit_time_sec", "mean"),
            predict_time_sec=("predict_time_sec", "mean"),
        )
    )


def _apply_baseline_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()

    baseline = (
        out[(out["reduction"] == "none") & (out["selection"] == "none")]
        .groupby(["track", "model"], as_index=False)
        .agg(
            baseline_accuracy=("accuracy", "max"),
            baseline_f1=("f1", "max"),
        )
    )

    out = out.merge(baseline, on=["track", "model"], how="left")
    out["delta_vs_baseline_accuracy"] = out["accuracy"] - out["baseline_accuracy"]
    out["delta_vs_baseline_f1"] = out["f1"] - out["baseline_f1"]
    return out


def build_rankings(summary: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    scored = _apply_baseline_deltas(summary)

    binary = (
        scored[scored["track"] == "binary"]
        .sort_values(["accuracy", "f1"], ascending=[False, False])
        .reset_index(drop=True)
    )
    binary.insert(0, "rank", range(1, len(binary) + 1))

    multiclass = (
        scored[scored["track"] == "multiclass"]
        .sort_values(["accuracy", "f1"], ascending=[False, False])
        .reset_index(drop=True)
    )
    multiclass.insert(0, "rank", range(1, len(multiclass) + 1))

    return binary, multiclass


def build_comparison_report(summary: pd.DataFrame) -> str:
    binary, multiclass = build_rankings(summary)

    def top_block(df: pd.DataFrame, label: str) -> str:
        cols = [
            "rank",
            "preprocessing",
            "reduction",
            "selection",
            "model",
            "accuracy",
            "f1",
            "delta_vs_baseline_accuracy",
        ]
        view = df.head(10)[cols].copy()
        for c in ["accuracy", "f1", "delta_vs_baseline_accuracy"]:
            view[c] = view[c].map(lambda v: f"{v:.4f}" if pd.notna(v) else "nan")
        return f"## {label} Top 10\n\n" + view.to_markdown(index=False)

    text = [
        "# Cartesian Comparison Report",
        "",
        top_block(binary, "Binary"),
        "",
        top_block(multiclass, "Multiclass"),
    ]
    return "\n".join(text)


def save_comparisons(summary: pd.DataFrame, tables_dir: Path, reports_dir: Path) -> Dict[str, Path]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    binary, multiclass = build_rankings(summary)

    binary_path = tables_dir / "cartesian_rankings_binary.csv"
    multiclass_path = tables_dir / "cartesian_rankings_multiclass.csv"
    report_path = reports_dir / "cartesian_comparison_report.md"

    binary.to_csv(binary_path, index=False)
    multiclass.to_csv(multiclass_path, index=False)
    report_path.write_text(build_comparison_report(summary), encoding="utf-8")

    return {
        "binary": binary_path,
        "multiclass": multiclass_path,
        "report": report_path,
    }
