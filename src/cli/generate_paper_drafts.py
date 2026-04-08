#!/usr/bin/env python3
"""Generate draft paper markdown sections from Cartesian benchmark outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DRAFT_DIR = ROOT / "paper" / "draft"
TABLES_DIR = ROOT / "paper" / "tables"

SUMMARY_PATH = ROOT / "results" / "tables" / "cartesian_summary_by_combo.csv"
BINARY_RANK_PATH = ROOT / "results" / "tables" / "cartesian_rankings_binary.csv"
MULTI_RANK_PATH = ROOT / "results" / "tables" / "cartesian_rankings_multiclass.csv"
MANIFEST_PATH = ROOT / "results" / "metrics" / "cartesian_run_manifest.json"

DRAFT_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def to_markdown_table(df: pd.DataFrame, max_rows: int = 12) -> str:
    if df.empty:
        return "_No data available yet._"
    return df.head(max_rows).to_markdown(index=False)


def _load_or_empty(path: Path, columns: list[str]) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=columns)


def main() -> None:
    summary = _load_or_empty(
        SUMMARY_PATH,
        [
            "track",
            "preprocessing",
            "reduction",
            "selection",
            "model",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
        ],
    )

    binary_rank = _load_or_empty(
        BINARY_RANK_PATH,
        [
            "rank",
            "preprocessing",
            "reduction",
            "selection",
            "model",
            "accuracy",
            "f1",
            "delta_vs_baseline_accuracy",
        ],
    )
    multi_rank = _load_or_empty(
        MULTI_RANK_PATH,
        [
            "rank",
            "preprocessing",
            "reduction",
            "selection",
            "model",
            "accuracy",
            "f1",
            "delta_vs_baseline_accuracy",
        ],
    )

    manifest = {}
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    expected_combos = int(manifest.get("expected_combos", 1536))
    expected_fold_evals = int(manifest.get("expected_fold_evals", 4608))

    best_binary = "TBD"
    best_multi = "TBD"
    if not binary_rank.empty:
        b = binary_rank.iloc[0]
        best_binary = (
            f"{b.get('model', 'NA')} | {b.get('preprocessing', 'NA')} | "
            f"{b.get('reduction', 'NA')} | {b.get('selection', 'NA')}"
        )
    if not multi_rank.empty:
        m = multi_rank.iloc[0]
        best_multi = (
            f"{m.get('model', 'NA')} | {m.get('preprocessing', 'NA')} | "
            f"{m.get('reduction', 'NA')} | {m.get('selection', 'NA')}"
        )

    (DRAFT_DIR / "01_abstract.md").write_text(
        "# Abstract\n\n"
        "This project presents a full soft-computing benchmark for epileptic seizure recognition using a "
        "staged Cartesian workflow. The pipeline evaluates preprocessing, feature reduction, feature selection, "
        "and classification under one reproducible protocol with two tracks (binary and multiclass). "
        f"The benchmark covers {expected_combos} unique method combinations and {expected_fold_evals} fold evaluations "
        "with robust skip/failure logging for invalid combinations.\n",
        encoding="utf-8",
    )

    (DRAFT_DIR / "02_introduction.md").write_text(
        "# Introduction\n\n"
        "This work builds a reproducible course-aligned framework for seizure classification, covering all requested "
        "soft-computing stages: preprocessing, reduction, selection, multi-model benchmarking, and paper-ready reporting.\n",
        encoding="utf-8",
    )

    (DRAFT_DIR / "03_related_work.md").write_text(
        "# Related Work\n\n"
        "Add at least 10 related studies with a table: Reference, Year, Methods, and Results.\n",
        encoding="utf-8",
    )

    (DRAFT_DIR / "04_methodology.md").write_text(
        "# Methodology\n\n"
        "## Pipeline\n"
        "1. Data loading and preprocessing\n"
        "2. Feature reduction (none, PCA, LDA projection, SVD)\n"
        "3. Feature selection (none, filter, wrapper, embedded, GA)\n"
        "4. Classification with six models\n"
        "5. 3-fold stratified CV and metric analysis\n",
        encoding="utf-8",
    )

    (DRAFT_DIR / "05_proposed_model.md").write_text(
        "# Proposed Model\n\n"
        "The proposed model is a staged Cartesian engine that enumerates all method combinations "
        "across tracks and folds, then logs metrics and skip reasons in a standardized schema.\n",
        encoding="utf-8",
    )

    bin_cols = [
        "rank",
        "preprocessing",
        "reduction",
        "selection",
        "model",
        "accuracy",
        "f1",
        "delta_vs_baseline_accuracy",
    ]
    multi_cols = [
        "rank",
        "preprocessing",
        "reduction",
        "selection",
        "model",
        "accuracy",
        "f1",
        "delta_vs_baseline_accuracy",
    ]

    results_md = (
        "# Results and Discussion\n\n"
        f"- Best binary combination (current run): **{best_binary}**\n"
        f"- Best multiclass combination (current run): **{best_multi}**\n\n"
        "## Binary Top Rankings\n\n"
        f"{to_markdown_table(binary_rank[bin_cols] if set(bin_cols).issubset(binary_rank.columns) else binary_rank)}\n\n"
        "## Multiclass Top Rankings\n\n"
        f"{to_markdown_table(multi_rank[multi_cols] if set(multi_cols).issubset(multi_rank.columns) else multi_rank)}\n\n"
        "## Notes\n"
        "- Discuss methods with highest F1 and accuracy gains over baseline (`none + none`).\n"
        "- Explain failed/skipped patterns using `status` and `skip_reason`.\n"
        "- Compare preprocessing/reduction/selection impacts by track.\n"
    )
    (DRAFT_DIR / "06_results_and_discussion.md").write_text(results_md, encoding="utf-8")

    (DRAFT_DIR / "07_conclusion_future_work.md").write_text(
        "# Conclusion and Future Work\n\n"
        "Summarize strongest combinations and propose next steps: stronger tuning, statistical testing, and "
        "cross-dataset validation.\n",
        encoding="utf-8",
    )

    (DRAFT_DIR / "08_references.md").write_text(
        "# References\n\n"
        "Add references in APA style and cite them in-text as [1], [2], ...\n",
        encoding="utf-8",
    )

    if not binary_rank.empty:
        binary_rank.head(20).to_csv(TABLES_DIR / "top_binary_results.csv", index=False)
    if not multi_rank.empty:
        multi_rank.head(20).to_csv(TABLES_DIR / "top_multiclass_results.csv", index=False)
    if not summary.empty:
        summary.to_csv(TABLES_DIR / "cartesian_summary_snapshot.csv", index=False)

    print(f"Draft sections generated in: {DRAFT_DIR}")


if __name__ == "__main__":
    main()
