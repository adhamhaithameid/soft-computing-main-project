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
Recent seizure-detection studies span classical machine learning, hybrid feature engineering, and deep-learning-assisted classifiers. Reported performance is often high on Bonn EEG and CHB-MIT variants, but direct comparison remains difficult because preprocessing, feature pipelines, and validation protocols vary across papers.

| Ref No. | Paper | Year | Methods | Reported Results |
|:--:|:--|:--:|:--|:--|
| [R1] | Siddiqui et al., *Brain Informatics* | 2020 | Review of ML classifiers for seizure detection | Summarized classifier trends and highlighted EEG noise/non-stationarity challenges. |
| [R2] | Liu et al., *Technology and Health Care* | 2017 | Temporal + wavelet features + kernel ELM | Reported satisfactory accuracy with lower computation time. |
| [R3] | Chen et al., *BMC Med Inform Decis Mak* | 2023 | DWT + entropy/STD feature fusion + RF feature selection + CNN | Bonn interictal/ictal: 99.9% accuracy; New Delhi interictal/ictal: 100% in their setup. |
| [R4] | Khalid et al., *DIGITAL HEALTH* | 2024 | ICA + prediction-probability features (FIR) + SVM | Reported 98.4% accuracy. |
| [R5] | Saranya & Bharathi, *JIFS* | 2024 | Hybrid IANFIS-LightGBM | Reported strongest classification performance among compared methods on Bonn and CHB-MIT records. |
| [R6] | Atlam et al., *Applied Sciences* | 2025 | PCA + DWT hybrid feature selection + SMOTE + SVM | Reported 97.30% accuracy, 99.62% AUC, 93.08% F1. |
| [R7] | Berrich & Guennoun, *Scientific Reports* | 2025 | PCA-based dimensionality reduction + CNN-SVM/DNN-SVM | Proposed hybrid deep+SVM EEG detection framework with dimensionality reduction. |
| [R8] | Chakrabarti et al., *JAISE* | 2022 | Moving-window pediatric seizure recognition with RF/DT/ANN/ensemble | RF best: 91.9% accuracy, 94.1% sensitivity, 89.7% specificity. |
| [R9] | Andrzejak et al., *Physical Review E* | 2001 | Nonlinear dynamics analysis of EEG brain states | Foundational EEG-state analysis used in building the widely used Bonn benchmark structure. |
| [R10] | Shoeb et al., *IEEE EMBC* | 2004 | Patient-specific seizure onset detection from EEG | Early influential patient-specific onset-detection framework. |

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
[R1] Siddiqui, M. K., Morales-Menendez, R., Huang, X., & Hussain, N. (2020). A review of epileptic seizure detection using machine learning classifiers. *Brain Informatics, 7*(1), Article 5. https://doi.org/10.1186/s40708-020-00105-1

[R2] Liu, Q., Zhao, X., Hou, Z., & Liu, H. (2017). Epileptic seizure detection based on the kernel extreme learning machine. *Technology and Health Care, 25*(S1), 399-409. https://doi.org/10.3233/THC-171343

[R3] Chen, W., Wang, Y., Ren, Y., Jiang, H., Du, G., Zhang, J., & Li, J. (2023). An automated detection of epileptic seizures EEG using CNN classifier based on feature fusion with high accuracy. *BMC Medical Informatics and Decision Making, 23*(1), Article 96. https://doi.org/10.1186/s12911-023-02180-w

[R4] Khalid, M., Raza, A., Akhtar, A., Rustam, F., Ballester, J. B., Rodriguez, C. L., Díez, I. T., & Ashraf, I. (2024). Diagnosing epileptic seizures using combined features from independent components and prediction probability from EEG data. *DIGITAL HEALTH, 10*. https://doi.org/10.1177/20552076241277185

[R5] Saranya, D., & Bharathi, A. (2024). Automatic detection of epileptic seizure using machine learning-based IANFIS-LightGBM system. *Journal of Intelligent & Fuzzy Systems, 46*(1), 2463-2482. https://doi.org/10.3233/JIFS-233430

[R6] Atlam, H. F., Aderibigbe, G. E., & Nadeem, M. S. (2025). Effective epileptic seizure detection with hybrid feature selection and SMOTE-based data balancing using SVM classifier. *Applied Sciences, 15*(9), 4690. https://doi.org/10.3390/app15094690

[R7] Berrich, Y., & Guennoun, Z. (2025). EEG-based epilepsy detection using CNN-SVM and DNN-SVM with feature dimensionality reduction by PCA. *Scientific Reports, 15*(1), Article 14313. https://doi.org/10.1038/s41598-025-95831-z

[R8] Chakrabarti, S., Swetapadma, A., & Pattnaik, P. K. (2022). An improved method for recognizing pediatric epileptic seizures based on advanced learning and moving window technique. *Journal of Ambient Intelligence and Smart Environments, 14*(1), 39-59. https://doi.org/10.3233/AIS-210042

[R9] Andrzejak, R. G., Lehnertz, K., Mormann, F., Rieke, C., David, P., & Elger, C. E. (2001). Indications of nonlinear deterministic and finite-dimensional structures in time series of brain electrical activity: Dependence on recording region and brain state. *Physical Review E, 64*(6). https://doi.org/10.1103/PhysRevE.64.061907

[R10] Shoeb, A., Edwards, H., Connolly, J., Bourgeois, B., Treves, T., & Guttag, J. (2004). Patient-specific seizure onset detection. In *The 26th Annual International Conference of the IEEE Engineering in Medicine and Biology Society* (Vol. 3, pp. 419-422). https://doi.org/10.1109/IEMBS.2004.1403183
