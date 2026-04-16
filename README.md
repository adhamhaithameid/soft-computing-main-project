# Epileptic Seizure Recognition - Phase 1 (CSC425)

Rubric-aligned Soft Computing Phase 1 project for epileptic seizure recognition using the UCI dataset.

## Open in Colab
- Colab notebook: https://colab.research.google.com/gist/adhamhaithameid/d775e807e6d0c3f2e44aa1655a2fdf3f/Phase1_colab.ipynb
- Source notebook file in repo: `Phase1_colab.ipynb`

## Project Files
- `Phase1.ipynb`: Main working notebook (source of truth)
- `Phase1_colab.ipynb`: Colab-ready notebook (clean metadata/outputs)
- `phase1.py`: Script version of the same pipeline
- `epileptic_seizure_data.csv`: Dataset file
- `Phase1_Research_Paper_Final.docx`: Final paper document

## Implemented Pipeline
1. Preprocessing
- Data visualization
- Missing values treatment
- Binning process
- Descriptive statistics
- Covariance/correlation/heatmap
- Chi-square, t-test, ANOVA

2. Feature Reduction and Selection
- PCA
- LDA projection
- SVD
- Kernel PCA (sample-based)
- Filter selection (SelectKBest)
- Wrapper selection (RFE)
- Embedded selection (RandomForest feature importance)

3. Models
- Naive Bayesian
- Bayesian Belief Network
- Decision Tree (Entropy)
- LDA Classifier
- Neural Network (Feed Forward)
- Feed Back Neural Network
- K-NN (Euclidean)
- K-NN (Manhattan)
- SVM (RBF Kernel)
- Logistic Regression

4. Evaluation
- 80/20 train-test split
- K-fold cross-validation
- Confusion matrix
- Accuracy, error rate, precision, recall, F1, ROC-AUC
- Overfitting/underfitting interpretation
- Regression metrics (MAE, RMSE, R2, Willmott, NSE, Legates-McCabe)

## Run Locally
```bash
python3 phase1.py
```

## Run in Notebook
1. Open `Phase1.ipynb` (or `Phase1_colab.ipynb` in Colab).
2. Run all cells from top to bottom.
3. Final run cell prints configured models and executes full pipeline.

## Notes
- The notebook defines `_overfit_interpretation` before model evaluation to avoid `NameError`.
- `Phase1.ipynb` and `Phase1_colab.ipynb` are synchronized.
