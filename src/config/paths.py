from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# New top-level structure
ASSETS_DIR = ROOT / "assets"
LECTURES_DIR = ASSETS_DIR / "lectures"

DATA_DIR = ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_INTERIM_DIR = DATA_DIR / "interim"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_CATALOG_DIR = DATA_DIR / "catalog"

NOTEBOOKS_DIR = ROOT / "notebooks"
NOTEBOOKS_COLAB_DIR = NOTEBOOKS_DIR / "colab"
NOTEBOOKS_KAGGLE_DIR = NOTEBOOKS_DIR / "kaggle"
NOTEBOOKS_LOCAL_DIR = NOTEBOOKS_DIR / "local"

SRC_DIR = ROOT / "src"
SRC_CLI_DIR = SRC_DIR / "cli"
SRC_CORE_DIR = SRC_DIR / "core"
SRC_CONFIG_DIR = SRC_DIR / "config"

RESULTS_DIR = ROOT / "results"
RESULTS_METRICS_DIR = RESULTS_DIR / "metrics"
RESULTS_TABLES_DIR = RESULTS_DIR / "tables"
RESULTS_FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_FOLDS_DIR = RESULTS_DIR / "folds"
RESULTS_REPORTS_DIR = RESULTS_DIR / "reports"

PAPER_DIR = ROOT / "paper"
PAPER_TEMPLATE_DIR = PAPER_DIR / "template"
PAPER_DRAFT_DIR = PAPER_DIR / "draft"
PAPER_TABLES_DIR = PAPER_DIR / "tables"
PAPER_FIGURES_DIR = PAPER_DIR / "figures"
PAPER_REFERENCES_DIR = PAPER_DIR / "references"

DOCS_DIR = ROOT / "docs"
DOCS_PLANS_DIR = DOCS_DIR / "plans"
DOCS_GUIDES_DIR = DOCS_DIR / "guides"
DOCS_STATUS_DIR = DOCS_DIR / "status"
DOCS_PAPER_DIR = DOCS_DIR / "paper"

# Project dataset defaults
DATASET_DIR = DATA_RAW_DIR / "epileptic_seizure_recognition"
DATASET_CSV = DATASET_DIR / "epileptic_seizure_data.csv"
DATASET_METADATA_JSON = DATASET_DIR / "metadata.json"


def ensure_structure() -> None:
    required = [
        LECTURES_DIR,
        DATA_RAW_DIR,
        DATA_INTERIM_DIR,
        DATA_PROCESSED_DIR,
        DATA_CATALOG_DIR,
        NOTEBOOKS_COLAB_DIR,
        NOTEBOOKS_KAGGLE_DIR,
        NOTEBOOKS_LOCAL_DIR,
        SRC_CLI_DIR,
        SRC_CORE_DIR,
        SRC_CONFIG_DIR,
        RESULTS_METRICS_DIR,
        RESULTS_TABLES_DIR,
        RESULTS_FIGURES_DIR,
        RESULTS_FOLDS_DIR,
        RESULTS_REPORTS_DIR,
        PAPER_TEMPLATE_DIR,
        PAPER_DRAFT_DIR,
        PAPER_TABLES_DIR,
        PAPER_FIGURES_DIR,
        PAPER_REFERENCES_DIR,
        DOCS_PLANS_DIR,
        DOCS_GUIDES_DIR,
        DOCS_STATUS_DIR,
        DOCS_PAPER_DIR,
    ]
    for path in required:
        path.mkdir(parents=True, exist_ok=True)
