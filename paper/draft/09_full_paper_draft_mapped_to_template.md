# Project Title
Epileptic Seizure Recognition Using Soft Computing Techniques: A Unified Pipeline for Preprocessing, Feature Engineering, and Classification

## Authors
- Student Name: [Fill Your Name]
- Course: Soft Computing (CSC425)
- Instructor: Dr. Ahmed Anter
- Semester: Spring 2026

## Abstract
This project presents an end-to-end soft computing workflow for epileptic seizure recognition using the Epileptic Seizure Recognition dataset. The workflow was designed to satisfy all required course stages in one reproducible pipeline: preprocessing, statistical analysis, feature reduction, feature selection, classification, evaluation, and paper-ready reporting. Two prediction tracks were used. The first track is a binary setup (seizure vs. non-seizure), and the second track is a multiclass setup with five classes. 

The preprocessing stage included numeric validation, missing-value checking and treatment, standardization, and descriptive statistical analysis. Data analysis artifacts were generated for minimum, maximum, mean, variance, standard deviation, skewness, kurtosis, covariance matrix, and correlation matrix. Feature engineering covered reduction methods (PCA, LDA, and SVD) and selection methods (filter ANOVA, wrapper SFS, embedded L1 regularization, and Genetic Algorithm feature subset search). Multiple classifiers were benchmarked on each feature-set branch, including KNN, SVM, Decision Tree, Logistic Regression, LDA classifier, and MLP-ANN. 

Evaluation used stratified K-fold cross-validation and confusion-matrix-driven metrics. On the full dataset (11,500 samples), the best binary performance was obtained by SVM with SVD features (accuracy = 0.9719, ROC-AUC = 0.9945). For multiclass classification, the best performance was obtained by MLP-ANN on original features (accuracy = 0.6747). Results demonstrate that strong performance can be achieved with careful feature engineering and model comparison inside a unified soft-computing pipeline.

## Keywords
Epileptic Seizure Recognition, Soft Computing, Feature Reduction, Feature Selection, Genetic Algorithm, SVM, MLP, Classification

## 1. Introduction
Epileptic seizure recognition is an important healthcare machine learning problem in which EEG-derived signals are analyzed to distinguish seizure and non-seizure patterns. Reliable prediction can support early detection and improve decision support for neurological analysis. In this project, we implemented a complete course-aligned pipeline covering all required phases from raw data handling to final quantitative comparison.

The main problem addressed is how to build a reproducible and fair comparison framework that includes preprocessing, feature engineering, and multiple classifiers, while tracking performance using consistent metrics. We applied the main soft computing methods covered in the course, including PCA, LDA, SVD, filter/wrapper/embedded selection strategies, and evolutionary optimization with Genetic Algorithms.

The key contribution of this project is a unified experimental framework where all methods are tested on the same data splits and evaluation protocol. This makes method-to-method comparison clear and defensible. In addition, the project outputs paper-ready tables and draft sections directly from experiment files, reducing reporting errors.

Paper organization is as follows: Section 2 presents related work structure, Section 3 explains methodology, Section 4 presents the proposed model pipeline, Section 5 reports results and discussion, and Section 6 concludes with future directions.

## 2. Related Work
This section should include at least 10 studies focused on seizure recognition or closely related EEG classification tasks. Use the following ready table format in your final DOCX and replace entries with your selected papers and exact reported accuracy values.

| Ref No. | Paper | Year | Methods | Reported Results |
|:--:|:--|:--:|:--|:--|
| [R1] | [Add seizure-related paper] | [Year] | [Methods] | [Accuracy/F1/AUC] |
| [R2] | [Add seizure-related paper] | [Year] | [Methods] | [Accuracy/F1/AUC] |
| [R3] | [Add seizure-related paper] | [Year] | [Methods] | [Accuracy/F1/AUC] |
| [R4] | [Add seizure-related paper] | [Year] | [Methods] | [Accuracy/F1/AUC] |
| [R5] | [Add seizure-related paper] | [Year] | [Methods] | [Accuracy/F1/AUC] |
| [R6] | [Add seizure-related paper] | [Year] | [Methods] | [Accuracy/F1/AUC] |
| [R7] | [Add seizure-related paper] | [Year] | [Methods] | [Accuracy/F1/AUC] |
| [R8] | [Add seizure-related paper] | [Year] | [Methods] | [Accuracy/F1/AUC] |
| [R9] | [Add seizure-related paper] | [Year] | [Methods] | [Accuracy/F1/AUC] |
| [R10] | [Add seizure-related paper] | [Year] | [Methods] | [Accuracy/F1/AUC] |

## 3. Methodology
### 3.1 Dataset Description
- Dataset: Epileptic Seizure Recognition
- Total samples: 11500
- Total predictive features: 178
- Target classes: 5 balanced classes
- Binary mapping: class 1 as seizure, classes 2-5 as non-seizure

### 3.2 Preprocessing and Statistical Analysis
- Numeric conversion and column validation were applied.
- Missing values were checked and treated (no unresolved missing values in the final run).
- Standardization was applied before model fitting.
- Statistical outputs were generated:
  - descriptive stats (min, max, mean, variance, std, skewness, kurtosis)
  - covariance matrix
  - correlation matrix

### 3.3 Feature Reduction Methods
- Principal Component Analysis (PCA)
- Linear Discriminant Analysis (LDA) reduction
- Singular Value Decomposition (SVD)

### 3.4 Feature Selection Methods
- Filter method: ANOVA SelectKBest
- Wrapper method: Sequential Forward Selection (SFS)
- Embedded method: L1-based selection
- Evolutionary method: Genetic Algorithm feature mask optimization

### 3.5 Classification Models
- KNN
- SVM
- Decision Tree
- Logistic Regression
- LDA Classifier
- MLP-ANN

### 3.6 Evaluation Protocol
- Stratified K-fold cross-validation
- Metrics: Accuracy, Error Rate, Precision, Recall, F1-score, ROC-AUC (binary)
- Confusion matrix for best model per track

## 4. Proposed Model
The proposed pipeline consists of the following phases:
1. Load and validate dataset
2. Preprocessing and statistical analysis
3. Parallel feature engineering branches:
   - reduction branch (PCA, LDA, SVD)
   - selection branch (filter, wrapper, embedded, GA)
   - baseline branch (original features)
4. Model training and evaluation for each branch
5. Ranking methods by cross-validated accuracy
6. Exporting tables, confusion matrices, and paper-ready markdown

This architecture enables direct comparison between all studied techniques under one reproducible implementation.

## 5. Results and Discussion
### 5.1 Summary of Experimental Coverage
- Total experiment combinations executed: 288
- Tracks: binary and multiclass
- Evaluation basis: cross-validated mean metrics

### 5.2 Binary Track Top Results
| feature_set   | model   |   accuracy |   precision |   recall |     f1 |   roc_auc |
|:--------------|:--------|-----------:|------------:|---------:|-------:|----------:|
| svd           | SVM     |     0.9719 |      0.9541 |   0.903  | 0.9279 |    0.9945 |
| original      | MLP_ANN |     0.9715 |      0.9527 |   0.9022 | 0.9267 |    0.9877 |
| pca           | SVM     |     0.9712 |      0.9543 |   0.8991 | 0.9259 |    0.9944 |
| original      | SVM     |     0.9707 |      0.9546 |   0.8961 | 0.9244 |    0.9946 |
| pca           | MLP_ANN |     0.9703 |      0.9426 |   0.9065 | 0.9242 |    0.9882 |
| svd           | MLP_ANN |     0.969  |      0.9362 |   0.9065 | 0.9211 |    0.9888 |
| embedded_l1   | MLP_ANN |     0.969  |      0.9562 |   0.8857 | 0.9192 |    0.9842 |
| embedded_l1   | SVM     |     0.9683 |      0.9519 |   0.8865 | 0.918  |    0.9946 |

Best binary configuration: **SVM + svd**
- Accuracy: **0.9719**
- Precision: **0.9541**
- Recall: **0.9030**
- F1-score: **0.9279**
- ROC-AUC: **0.9945**

### 5.3 Multiclass Track Top Results
| feature_set   | model   |   accuracy |   precision |   recall |     f1 |
|:--------------|:--------|-----------:|------------:|---------:|-------:|
| original      | MLP_ANN |     0.6747 |      0.6768 |   0.6747 | 0.6755 |
| svd           | MLP_ANN |     0.6713 |      0.6725 |   0.6713 | 0.6718 |
| pca           | MLP_ANN |     0.6709 |      0.6718 |   0.6709 | 0.6711 |
| embedded_l1   | MLP_ANN |     0.662  |      0.6636 |   0.662  | 0.6624 |
| ga_selection  | MLP_ANN |     0.6317 |      0.6364 |   0.6317 | 0.6331 |
| filter_kbest  | MLP_ANN |     0.5727 |      0.5743 |   0.5727 | 0.5728 |
| pca           | SVM     |     0.5504 |      0.5968 |   0.5504 | 0.5324 |
| svd           | SVM     |     0.5483 |      0.594  |   0.5483 | 0.5304 |

Best multiclass configuration: **MLP_ANN + original**
- Accuracy: **0.6747**
- Precision: **0.6768**
- Recall: **0.6747**
- F1-score: **0.6755**

### 5.4 Interpretation
- Binary performance is significantly stronger than multiclass performance, which is expected because the binary separation task is simpler.
- SVM with SVD achieved the strongest binary score, showing that reduced representations can preserve discriminative information while controlling noise.
- MLP-ANN performed best in multiclass classification, suggesting nonlinear modeling capacity is useful when class boundaries are more complex.
- Wrapper and GA-based feature selection provided competitive but not always top performance, while demanding more computation.
- The results confirm that model performance depends strongly on the joint choice of feature engineering and classifier, not classifier choice alone.

### 5.5 Overfitting/Underfitting Discussion
- The gap between binary and multiclass tracks indicates increased multiclass difficulty rather than direct model failure.
- High binary AUC alongside strong F1 supports robust ranking and classification behavior in the seizure-vs-non-seizure task.
- To further diagnose overfitting, future runs should add train-vs-validation curve plots for top models and per-fold variance summaries.

## 6. Conclusion and Future Work
This project delivered a full soft-computing workflow for epileptic seizure recognition using a reproducible and paper-oriented pipeline. All major course requirements were implemented in one framework: preprocessing, feature reduction, feature selection, genetic optimization, classifier benchmarking, and metric-driven evaluation. The strongest binary configuration was SVM with SVD, while MLP-ANN gave the best multiclass performance.

Future work directions include: (1) extending feature engineering with domain-specific EEG transformations, (2) hyperparameter optimization using Bayesian search, (3) deeper neural architectures for multiclass improvement, (4) richer visual analytics for fold-wise behavior and calibration, and (5) external validation on additional seizure datasets to test generalization.

## 7. References (Starter)
Use APA style in the final DOCX. The following list is a starter and can be extended with seizure-specific studies in Section 2.

[1] Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems. *Annals of Eugenics*, 7(2), 179-188.

[2] Hotelling, H. (1933). Analysis of a complex of statistical variables into principal components. *Journal of Educational Psychology*, 24(6), 417-441.

[3] Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. *IEEE Transactions on Information Theory*, 13(1), 21-27.

[4] Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20, 273-297.

[5] Quinlan, J. R. (1986). Induction of decision trees. *Machine Learning*, 1, 81-106.

[6] Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society: Series B*, 58(1), 267-288.

[7] Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems*. University of Michigan Press.

[8] Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323, 533-536.

[9] Andrzejak, R. G., Lehnertz, K., Mormann, F., Rieke, C., David, P., & Elger, C. E. (2001). Indications of nonlinear deterministic and finite-dimensional structures in time series of brain electrical activity. *Physical Review E*, 64(6), 061907.

[10] Dua, D., & Graff, C. (2019). UCI Machine Learning Repository. University of California, Irvine, School of Information and Computer Sciences.
