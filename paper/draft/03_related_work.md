# Related Work

Recent seizure-detection studies span classical machine learning, hybrid feature engineering, and deep-learning-assisted classifiers. Most strong results are reported on Bonn EEG and CHB-MIT variants, but studies differ in preprocessing, feature extraction, and evaluation protocol, which makes direct comparison difficult.

| Ref | Year | Study | Main Methods | Reported Results |
|:--:|:--:|:--|:--|:--|
| R1 | 2020 | Siddiqui et al. | Systematic review of ML classifiers for seizure detection | Summarized major ML pipelines and highlighted noise/non-stationarity/data-volume challenges in EEG. |
| R2 | 2017 | Liu et al. | Temporal + wavelet features + kernel ELM | Reported satisfactory accuracy with lower computation time than baseline approaches. |
| R3 | 2023 | Chen et al. | DWT + entropy/STD feature fusion + RF feature selection + CNN | Bonn: 99.9% accuracy (interictal vs ictal), New Delhi: 100% accuracy in reported setup. |
| R4 | 2024 | Khalid et al. | ICA + prediction-probability features (FIR) + SVM | Reported 98.4% accuracy for epileptic seizure detection. |
| R5 | 2024 | Saranya & Bharathi | Hybrid IANFIS-LightGBM | Reported strongest classification performance among compared methods on Bonn and CHB-MIT records. |
| R6 | 2025 | Atlam et al. | PCA + DWT hybrid feature selection + SMOTE + SVM | Reported 97.30% accuracy, 99.62% AUC, and 93.08% F1. |
| R7 | 2025 | Berrich & Guennoun | PCA-based dimensionality reduction + CNN-SVM/DNN-SVM | Proposed hybrid deep+SVM framework for EEG epilepsy detection on reduced features. |
| R8 | 2022 | Chakrabarti et al. | Moving-window pediatric seizure recognition with RF/DT/ANN/ensemble | RF performed best with 91.9% accuracy, 94.1% sensitivity, 89.7% specificity. |
| R9 | 2001 | Andrzejak et al. | Nonlinear dynamics analysis of EEG brain states | Foundational EEG-state study used to build the widely adopted Bonn benchmark data structure. |
| R10 | 2004 | Shoeb et al. | Patient-specific seizure onset detection from EEG | Early influential patient-specific detection framework for real-time seizure onset studies. |
