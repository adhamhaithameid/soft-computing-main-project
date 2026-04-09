# Project Title
Epileptic Seizure Recognition with a Full Cartesian Soft Computing Benchmark

## Authors
- Student Name: [Fill Your Name]
- Course: Soft Computing (CSC425)
- Instructor: Dr. Ahmed Anter
- Semester: Spring 2026

## Abstract
This project presents an end-to-end soft computing workflow for epileptic seizure recognition using a fully refactored and reproducible Cartesian benchmark pipeline. The workflow covers all required stages: preprocessing, feature reduction, feature selection, classification, evaluation, visualization, and paper-ready reporting.

Two prediction tracks are evaluated: binary seizure detection and multiclass classification. The benchmark combines four preprocessing methods, four reduction options, eight selection strategies, and six classifiers under three-fold stratified cross-validation. This yields 1,536 unique method combinations and 4,608 fold-level evaluations.

Invalid combinations are handled safely through auto-fix and skip logging (`status`, `skip_reason`) so execution continues and coverage remains auditable. Final outputs include per-combination metrics, ranking tables, baseline deltas, comparison reports, and figure suites (heatmaps, top-N bars, fold-variance, ROC for top binary pipelines).

## Keywords
Soft Computing, Epileptic Seizure Recognition, Cartesian Benchmark, Feature Reduction, Feature Selection, Genetic Algorithm, Classification

## 1. Introduction
Epileptic seizure recognition is a high-impact classification problem where robust machine learning can support decision making in neurological analysis. A key challenge in course projects is not only obtaining good accuracy, but proving fair and reproducible comparisons across many techniques.

To address this, the project was redesigned as a Cartesian benchmark framework where each method family is explicitly enumerated and evaluated under a common protocol. This allows direct, transparent comparison between preprocessing, reduction, selection, and model choices.

## 2. Related Work
Add at least 10 seizure/EEG classification studies with method and metric comparisons.

| Ref No. | Paper | Year | Methods | Reported Results |
|:--:|:--|:--:|:--|:--|
| [R1] | [Add paper] | [Year] | [Methods] | [Accuracy/F1/AUC] |
| [R2] | [Add paper] | [Year] | [Methods] | [Accuracy/F1/AUC] |
| [R3] | [Add paper] | [Year] | [Methods] | [Accuracy/F1/AUC] |
| [R4] | [Add paper] | [Year] | [Methods] | [Accuracy/F1/AUC] |
| [R5] | [Add paper] | [Year] | [Methods] | [Accuracy/F1/AUC] |
| [R6] | [Add paper] | [Year] | [Methods] | [Accuracy/F1/AUC] |
| [R7] | [Add paper] | [Year] | [Methods] | [Accuracy/F1/AUC] |
| [R8] | [Add paper] | [Year] | [Methods] | [Accuracy/F1/AUC] |
| [R9] | [Add paper] | [Year] | [Methods] | [Accuracy/F1/AUC] |
| [R10] | [Add paper] | [Year] | [Methods] | [Accuracy/F1/AUC] |

## 3. Methodology
### 3.1 Dataset
- Dataset: Epileptic Seizure Recognition
- Samples: 11,500
- Features: 178 numeric predictors
- Targets:
  - Binary track: class 1 vs classes 2-5
  - Multiclass track: original five classes

### 3.2 Preprocessing
- `standard`
- `minmax`
- `robust`
- `quantile`

### 3.3 Feature Reduction
- `none`
- `pca`
- `lda_projection`
- `svd`

### 3.4 Feature Selection
- `none`
- `filter_chi2`
- `filter_anova`
- `filter_correlation`
- `wrapper_sfs`
- `wrapper_rfe`
- `embedded_l1`
- `ga_selection`

### 3.5 Classifiers
- `knn`
- `svm`
- `decision_tree`
- `logistic_regression`
- `lda_classifier`
- `mlp_ann`

### 3.6 Evaluation Protocol
- 3-fold stratified CV
- Metrics: accuracy, precision, recall, f1, roc_auc (binary), error_rate
- Runtime metrics: fit and prediction time
- Failure handling: logged per fold with `status` and `skip_reason`

## 4. Proposed Model
The project uses a deterministic staged Cartesian engine:
1. Load and clean data.
2. For each track and fold, apply each preprocessing method.
3. Apply each reduction method.
4. Apply each selection method.
5. Train/evaluate each classifier.
6. Save fold-level metrics and status rows.
7. Aggregate rankings, baseline deltas, and plots.

Combination math:
- Unique combos = `4 x 4 x 8 x 6 x 2 = 1536`
- Fold evaluations = `1536 x 3 = 4608`

## 5. Results and Discussion
### 5.1 Required result files
- `results/metrics/cartesian_metrics_all.csv`
- `results/metrics/cartesian_run_manifest.json`
- `results/tables/cartesian_summary_by_combo.csv`
- `results/tables/cartesian_rankings_binary.csv`
- `results/tables/cartesian_rankings_multiclass.csv`
- `results/reports/cartesian_comparison_report.md`

### 5.2 Full Run Snapshot (April 9, 2026, M1 CPU)
- Total fold evaluations: `4608 / 4608`
- Successful rows: `4392`
- Skipped/failed rows: `216`
- Runtime: `5294.26 sec` (`88.24 min`)

Best pipelines from `cartesian_run_manifest.json`:
- Binary: `svm + quantile + pca + none`  
  `accuracy=0.976261`, `precision=0.960050`, `recall=0.919563`, `f1=0.939349`, `roc_auc=0.995438`
- Multiclass: `mlp_ann + minmax + pca + none`  
  `accuracy=0.685651`, `precision=0.685624`, `recall=0.685660`, `f1=0.685026`

### 5.3 Top-Ranked Pipelines (Accuracy)
Binary track (top 5):

| Rank | Preprocessing | Reduction | Selection | Model | Accuracy | F1 | Delta vs baseline accuracy |
|:---:|:---|:---|:---|:---|---:|---:|---:|
| 1 | quantile | pca | none | svm | 0.976261 | 0.939349 | +0.000261 |
| 2 | quantile | none | none | svm | 0.976000 | 0.938647 | +0.000000 |
| 3 | quantile | svd | none | svm | 0.974608 | 0.934881 | -0.001391 |
| 4 | quantile | none | embedded_l1 | svm | 0.973565 | 0.932277 | -0.002435 |
| 5 | standard | svd | none | svm | 0.971913 | 0.927855 | -0.004087 |

Multiclass track (top 5):

| Rank | Preprocessing | Reduction | Selection | Model | Accuracy | F1 | Delta vs baseline accuracy |
|:---:|:---|:---|:---|:---|---:|---:|---:|
| 1 | minmax | pca | none | mlp_ann | 0.685651 | 0.685026 | +0.010956 |
| 2 | standard | none | none | mlp_ann | 0.674696 | 0.675497 | +0.000000 |
| 3 | standard | svd | none | mlp_ann | 0.671305 | 0.671783 | -0.003390 |
| 4 | standard | pca | none | mlp_ann | 0.670869 | 0.671112 | -0.003826 |
| 5 | robust | none | none | mlp_ann | 0.669740 | 0.670368 | -0.004956 |

### 5.4 Failure and Skip Analysis
All `216` non-OK rows were safe failures in feature selection edge cases:
- `72`: SequentialFeatureSelector requires at least 2 features (`shape=(700, 1)` after prior stages).
- `72`: `n_features_to_select` must be strictly less than available features.
- `48`: RFE requires at least 2 features (`shape=(7667, 1)`).
- `24`: RFE requires at least 2 features (`shape=(7666, 1)`).

This pattern indicates the failure handling behaved as intended: invalid low-dimensional combinations were logged and skipped without stopping the full benchmark.

### 5.5 Discussion
- Binary performance was dominated by SVM variants, especially with `quantile` preprocessing and no extra selection.
- Multiclass performance was dominated by MLP/ANN, with a clear gain from `minmax + pca` over the multiclass baseline.
- Wrapper/embedded selectors did not consistently beat the `none` selection baseline in top-ranked rows for this dataset, suggesting that strong preprocessing/reduction choices contributed more than aggressive selection in this run.

## 6. Conclusion and Future Work
This project delivers a complete, reproducible soft-computing benchmark that evaluates all requested method families in one framework. The refactor improves traceability, output organization, and experimental rigor by enforcing full Cartesian accounting and standardized schemas.

Future work:
1. Add hyperparameter optimization for top-ranked combinations.
2. Expand to additional datasets from the course list.
3. Add statistical significance tests between top pipelines.

## 7. References
Use APA style and map citations consistently between text and bibliography.
