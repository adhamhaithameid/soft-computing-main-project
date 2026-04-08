# Folder Structure

```text
soft computing - research/
├── assets/
│   └── lectures/
├── data/
│   ├── raw/
│   │   └── epileptic_seizure_recognition/
│   │       ├── epileptic_seizure_data.csv
│   │       └── metadata.json
│   ├── interim/
│   ├── processed/
│   └── catalog/
│       └── links.md
├── notebooks/
│   ├── colab/
│   │   ├── epileptic_seizure_full_pipeline_colab.ipynb
│   │   └── README.md
│   ├── kaggle/
│   └── local/
├── src/
│   ├── config/
│   │   └── paths.py
│   ├── core/
│   │   ├── cartesian_pipeline.py
│   │   ├── benchmark.py
│   │   ├── comparisons.py
│   │   ├── plots.py
│   │   └── runner.py
│   └── cli/
│       ├── fetch_data.py
│       ├── check_env.py
│       ├── run_experiments.py
│       └── generate_paper_drafts.py
├── results/
│   ├── metrics/
│   ├── tables/
│   ├── figures/
│   ├── folds/
│   └── reports/
├── paper/
│   ├── template/
│   ├── draft/
│   ├── tables/
│   ├── figures/
│   └── references/
├── docs/
│   ├── plans/
│   ├── guides/
│   ├── status/
│   └── paper/
├── README.md
├── PROJECT_MASTER_GUIDE.md
├── FOLDER_STRUCTURE.md
├── ABOUT.md
├── requirements.txt
├── run_all.sh
└── .gitignore
```

## Root Policy
Root keeps only top-level guides + run files:
- `README.md`
- `PROJECT_MASTER_GUIDE.md`
- `FOLDER_STRUCTURE.md`
- `ABOUT.md`
- `requirements.txt`
- `run_all.sh`
- `.gitignore`

All detailed docs are under `docs/*`.
