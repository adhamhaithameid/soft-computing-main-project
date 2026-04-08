#!/usr/bin/env python3
"""Generate draft paper markdown sections from experiment outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DRAFT_DIR = ROOT / "06_paper" / "draft"
TABLES_DIR = ROOT / "06_paper" / "tables"
METRICS_SUMMARY = ROOT / "05_results" / "tables" / "summary_accuracy.csv"

DRAFT_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def to_markdown_table(df: pd.DataFrame, max_rows: int = 12) -> str:
    if df.empty:
        return "_No data available yet._"
    return df.head(max_rows).to_markdown(index=False)


def main() -> None:
    if METRICS_SUMMARY.exists():
        summary = pd.read_csv(METRICS_SUMMARY)
    else:
        summary = pd.DataFrame(columns=["track", "feature_set", "model", "accuracy", "precision", "recall", "f1", "roc_auc"])

    top_binary = summary[summary["track"] == "binary"].sort_values("accuracy", ascending=False)
    top_multi = summary[summary["track"] == "multiclass"].sort_values("accuracy", ascending=False)

    (DRAFT_DIR / "01_abstract.md").write_text(
        "# Abstract\n\n"
        "This project investigates soft computing techniques for epileptic seizure recognition using two prediction tracks: binary seizure detection and multiclass classification. "
        "We apply preprocessing, feature reduction (PCA/LDA/SVD), feature selection (filter, wrapper, embedded, and genetic algorithm), and multiple classifiers (KNN, SVM, Decision Tree, Logistic Regression, LDA classifier, and ANN/MLP). "
        "Results are evaluated with 5-fold cross-validation, confusion matrices, and standard classification metrics to compare performance across methods.\n",
        encoding="utf-8",
    )

    (DRAFT_DIR / "02_introduction.md").write_text(
        "# Introduction\n\n"
        "Epileptic seizure recognition is an important pattern-recognition problem in healthcare. In this project, we build a full-course soft computing workflow that integrates dimensionality reduction, feature selection, genetic optimization, and supervised classification models. "
        "The main contribution is a unified reproducible pipeline that compares all required methods under the same dataset and evaluation protocol.\n",
        encoding="utf-8",
    )

    (DRAFT_DIR / "03_related_work.md").write_text(
        "# Related Work\n\n"
        "Add at least 10 related studies here with a comparison table containing: Reference, Year, Methods, and Results.\n",
        encoding="utf-8",
    )

    (DRAFT_DIR / "04_methodology.md").write_text(
        "# Methodology\n\n"
        "## Pipeline\n"
        "1. Data loading and preprocessing\n"
        "2. Feature reduction (PCA, LDA, SVD)\n"
        "3. Feature selection (filter, wrapper, embedded, GA)\n"
        "4. Classification with six models\n"
        "5. 5-fold cross-validation and metric analysis\n",
        encoding="utf-8",
    )

    (DRAFT_DIR / "05_proposed_model.md").write_text(
        "# Proposed Model\n\n"
        "Describe each phase of the implemented workflow and include the system diagram.\n",
        encoding="utf-8",
    )

    results_md = (
        "# Results and Discussion\n\n"
        "## Binary Track Top Results\n\n"
        f"{to_markdown_table(top_binary[['feature_set', 'model', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']])}\n\n"
        "## Multiclass Track Top Results\n\n"
        f"{to_markdown_table(top_multi[['feature_set', 'model', 'accuracy', 'precision', 'recall', 'f1']])}\n\n"
        "## Notes\n"
        "- Discuss preprocessing effects and statistical analysis.\n"
        "- Compare PCA/LDA/SVD performance.\n"
        "- Compare filter/wrapper/embedded/GA feature selection outcomes.\n"
        "- Discuss overfitting/underfitting patterns from fold behavior and confusion matrices.\n"
    )
    (DRAFT_DIR / "06_results_and_discussion.md").write_text(results_md, encoding="utf-8")

    (DRAFT_DIR / "07_conclusion_future_work.md").write_text(
        "# Conclusion and Future Work\n\n"
        "Summarize the best-performing methods and findings, then propose future improvements such as advanced deep models, feature engineering, and external validation datasets.\n",
        encoding="utf-8",
    )

    (DRAFT_DIR / "08_references.md").write_text(
        "# References\n\n"
        "Add references in APA style and cite them in-text as [1], [2], ...\n",
        encoding="utf-8",
    )

    # Export top-result tables separately for direct paper insertion
    if not top_binary.empty:
        top_binary.head(15).to_csv(TABLES_DIR / "top_binary_results.csv", index=False)
    if not top_multi.empty:
        top_multi.head(15).to_csv(TABLES_DIR / "top_multiclass_results.csv", index=False)

    print(f"Draft sections generated in: {DRAFT_DIR}")


if __name__ == "__main__":
    main()
