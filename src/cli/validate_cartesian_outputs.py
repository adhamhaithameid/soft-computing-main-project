#!/usr/bin/env python3
"""Validate Cartesian benchmark outputs for integrity and schema completeness."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.paths import (
    RESULTS_FIGURES_DIR,
    RESULTS_METRICS_DIR,
    RESULTS_REPORTS_DIR,
    RESULTS_TABLES_DIR,
)
from src.core import CartesianSpec

REQUIRED_METRICS_COLUMNS = [
    "track",
    "fold",
    "preprocessing",
    "reduction",
    "selection",
    "model",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "error_rate",
    "fit_time_sec",
    "predict_time_sec",
    "status",
    "skip_reason",
]


def _check_method_coverage(df: pd.DataFrame, spec: CartesianSpec) -> List[str]:
    errors: List[str] = []
    coverage = {
        "track": ("track", set(spec.tracks)),
        "preprocessing": ("preprocessing", set(spec.preprocessing)),
        "reduction": ("reduction", set(spec.reduction)),
        "selection": ("selection", set(spec.selection)),
        "model": ("model", set(spec.classifiers)),
    }
    for label, (col, expected) in coverage.items():
        seen = set(df[col].dropna().astype(str).tolist())
        missing = sorted(expected - seen)
        if missing:
            errors.append(f"Missing {label} coverage values: {', '.join(missing)}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow partial run counts (useful for smoke tests with --max-rows).",
    )
    args = parser.parse_args()

    spec = CartesianSpec(cv_splits=3)

    metrics_csv = RESULTS_METRICS_DIR / "cartesian_metrics_all.csv"
    manifest_json = RESULTS_METRICS_DIR / "cartesian_run_manifest.json"

    required_outputs = [
        metrics_csv,
        manifest_json,
        RESULTS_TABLES_DIR / "cartesian_summary_by_combo.csv",
        RESULTS_TABLES_DIR / "cartesian_rankings_binary.csv",
        RESULTS_TABLES_DIR / "cartesian_rankings_multiclass.csv",
        RESULTS_REPORTS_DIR / "cartesian_comparison_report.md",
    ]

    errors: List[str] = []
    notes: List[str] = []

    for p in required_outputs:
        if not p.exists():
            errors.append(f"Missing required output: {p}")

    figure_paths = sorted(RESULTS_FIGURES_DIR.glob("cartesian_*.png"))
    if not figure_paths:
        notes.append("No cartesian figures found under results/figures (plotting deps may be missing).")

    if errors:
        print("Validation failed before data checks:")
        for e in errors:
            print(f"- {e}")
        raise SystemExit(1)

    df = pd.read_csv(metrics_csv)
    missing_cols = [c for c in REQUIRED_METRICS_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing metrics schema columns: {', '.join(missing_cols)}")

    with manifest_json.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    if manifest.get("expected_combos") != spec.expected_combos:
        errors.append(
            f"Manifest expected_combos mismatch: got {manifest.get('expected_combos')} expected {spec.expected_combos}"
        )
    if manifest.get("expected_fold_evals") != spec.expected_fold_evals:
        errors.append(
            f"Manifest expected_fold_evals mismatch: got {manifest.get('expected_fold_evals')} expected {spec.expected_fold_evals}"
        )

    completed_ok = int(manifest.get("completed_ok", -1))
    skipped_or_failed = int(manifest.get("skipped_or_failed", -1))
    total_reported = completed_ok + skipped_or_failed

    if args.allow_partial:
        if total_reported > spec.expected_fold_evals:
            errors.append(
                f"Reported rows exceed expected fold evals: {total_reported} > {spec.expected_fold_evals}"
            )
    else:
        if total_reported != spec.expected_fold_evals:
            errors.append(
                f"Manifest count mismatch: completed_ok + skipped_or_failed = {total_reported}, "
                f"expected {spec.expected_fold_evals}"
            )
        if len(df) != spec.expected_fold_evals:
            errors.append(f"Metrics row count mismatch: {len(df)} != {spec.expected_fold_evals}")

    if "status" in df.columns:
        bad_status = sorted(set(df["status"].astype(str)) - {"ok", "failed"})
        if bad_status:
            errors.append(f"Unexpected status values: {', '.join(bad_status)}")

    ok = df[df["status"] == "ok"].copy() if "status" in df.columns else pd.DataFrame()
    for metric in ["accuracy", "precision", "recall", "f1"]:
        if metric in ok.columns and not ok.empty:
            vals = pd.to_numeric(ok[metric], errors="coerce")
            out_of_range = vals[(vals < 0) | (vals > 1)]
            if not out_of_range.empty:
                errors.append(f"Out-of-range values in {metric}: {len(out_of_range)} rows")

    if not df.empty:
        coverage_issues = _check_method_coverage(df, spec)
        if args.allow_partial:
            notes.extend(coverage_issues)
        else:
            errors.extend(coverage_issues)
    else:
        errors.append("Metrics CSV is empty.")

    if errors:
        print("Validation failed:")
        for e in errors:
            print(f"- {e}")
        for n in notes:
            print(f"- NOTE: {n}")
        raise SystemExit(1)

    report_lines = [
        "# Cartesian Validation Report",
        "",
        f"- Timestamp (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"- Expected combos: {spec.expected_combos}",
        f"- Expected fold evals: {spec.expected_fold_evals}",
        f"- Metrics rows: {len(df)}",
        f"- Completed ok: {completed_ok}",
        f"- Skipped or failed: {skipped_or_failed}",
        f"- Figures found: {len(figure_paths)}",
        f"- Mode: {'partial allowed' if args.allow_partial else 'strict full run'}",
        "",
        "Validation result: PASS",
    ]
    if notes:
        report_lines.extend(["", "## Notes"])
        report_lines.extend([f"- {n}" for n in notes])

    RESULTS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_REPORTS_DIR / "cartesian_validation_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("Validation passed.")
    print(f"Report written: {report_path}")


if __name__ == "__main__":
    main()
