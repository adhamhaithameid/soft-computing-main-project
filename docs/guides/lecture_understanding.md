# Lecture Understanding Summary

This file summarizes what was understood from the provided lecture PDFs and how each part maps to the project pipeline.

## 1) ANN (Lecture 1-2)
- Soft computing focuses on approximation, uncertainty, and robustness rather than strict exactness.
- Core ANN ideas include neuron model, weighted sum, activation functions, and learning paradigms.
- Learning paradigms covered: supervised and unsupervised.
- Mentioned supervised algorithms include perceptron, backpropagation, MLP, and RBF networks.
- Practical implication for this project: ANN/MLP is included as one of the classifiers for seizure prediction.

## 2) PCA (Lecture 2)
- PCA is an unsupervised dimensionality reduction method.
- It converts correlated features to linearly uncorrelated principal components.
- It is based on covariance structure and eigenvalues/eigenvectors.
- Main use: reduce dimensionality while preserving most variance.
- Practical implication for this project: PCA is used as a feature reduction branch before classification.

## 3) Decision Tree + Classification Metrics (Lecture 3)
- Decision trees are supervised methods for classification and regression.
- Splitting concepts include entropy and information gain (ID3/C4.5 style ideas).
- Tree-building is greedy and can become complex with high-dimensional data.
- Practical implication for this project: Decision Tree is included as a baseline/interpretable classifier.

## 4) KNN (Lecture 4)
- KNN is an instance-based, non-parametric method using neighborhood similarity.
- Distance choices matter (Euclidean/Manhattan/Hamming depending on variable type).
- Feature scaling/normalization is important for distance-based methods.
- Practical implication for this project: KNN is used after standardization as a core baseline model.

## 5) LDA (Lecture 4)
- LDA is supervised and can be used for both dimensionality reduction and classification.
- The objective is maximizing between-class separation while minimizing within-class scatter.
- Practical implication for this project:
  - LDA reduction is included as a reduction method.
  - LDA classifier is also included in model comparisons.

## 6) Feature Selection Strategies (Lecture 5)
- Three groups are emphasized:
  - Filter methods (e.g., ANOVA, correlation, chi-square).
  - Wrapper methods (e.g., forward/recursive selection, SFS).
  - Embedded methods (e.g., L1/L2-regularized models).
- Tradeoff: filter is fast, wrapper is computationally expensive but can perform better, embedded balances both.
- Practical implication for this project:
  - `filter_anova` and `filter_chi2` implemented.
  - `wrapper_sfs` implemented.
  - `embedded_l1` implemented.

## 7) Genetic Algorithms (Lecture 6)
- GA is an evolutionary optimization approach with population-based search.
- Core operators: selection, crossover, mutation.
- Uses a fitness function to evaluate candidate solutions.
- Practical implication for this project: GA-based feature subset selection is implemented as `ga_selection`.

## 8) Course-to-Project Mapping
- Preprocessing: numeric conversion, missing-value handling, scaling.
- Feature reduction: PCA, LDA, SVD.
- Feature selection: filter, wrapper (SFS), embedded (L1), GA.
- Models: KNN, SVM, Decision Tree, Logistic Regression, LDA classifier, MLP ANN.
- Evaluation: accuracy, precision, recall, F1, ROC-AUC (binary), confusion matrices, fold-wise metrics.

## 9) Current dataset decision justification
- Chosen dataset: **Epileptic Seizure Recognition**.
- Reason for choosing as easiest workable option for course scope:
  - Tabular format (simple preprocessing).
  - Clear classification target (supports binary + multiclass experiments).
  - Compatible with all lecture methods without heavy image/signal preprocessing pipelines.
