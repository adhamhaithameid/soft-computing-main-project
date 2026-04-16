"""
Phase 1 Soft Computing Pipeline (Epileptic Dataset, Binary Target)

Dataset explanation:
- Source file: epileptic_seizure_data.csv
- Shape: 11,500 rows, 180 columns in the raw file
- Columns: unnamed index-like column + X1..X178 features + y label
- Original labels: y in {1,2,3,4,5}, balanced at 2,300 rows per class
- Binary mapping used in this migrated project:
  - Positive class (Seizure)     -> y == 1
  - Negative class (Non-seizure) -> y in {2,3,4,5}

Why this mapping:
- It preserves the prior project convention for seizure-vs-non-seizure analysis,
  while keeping the compact "new stuff" project structure and workflow style.
"""

import pandas as pd
import urllib.request
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SKLDA
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    f1_score,
)
from scipy import stats

# --- CELL 1: Data Loading & Target Encoding ---

sns.set_style("whitegrid")
DATA_PATH = "epileptic_seizure_data.csv"
DATA_URL_CANDIDATES = [
    "https://raw.githubusercontent.com/adhamhaithameid/epileptic-seizure-recognition/experimental/new-stuff-migration/epileptic_seizure_data.csv",
    "https://raw.githubusercontent.com/adhamhaithameid/epileptic-seizure-recognition/main/epileptic_seizure_data.csv",
    "https://raw.githubusercontent.com/akshayg056/Epileptic-seizure-detection-/master/data.csv",
]

if not os.path.exists(DATA_PATH):
    print(f"Local dataset not found: {DATA_PATH}. Attempting download...")
    downloaded = False
    for url in DATA_URL_CANDIDATES:
        try:
            urllib.request.urlretrieve(url, DATA_PATH)
            print(f"Downloaded dataset from: {url}")
            downloaded = True
            break
        except Exception as exc:
            print(f"Failed URL: {url} -> {exc}")

    if not downloaded:
        print("Error: unable to find or download epileptic_seizure_data.csv")
        raise SystemExit(1)

df = pd.read_csv(DATA_PATH)

# Drop unnamed index-like columns (e.g., '', 'Unnamed: 0').
unnamed_cols = [c for c in df.columns if c == "" or str(c).startswith("Unnamed")]
if unnamed_cols:
    df = df.drop(columns=unnamed_cols)

target_col = "y" if "y" in df.columns else df.columns[-1]
X = df.drop(columns=[target_col]).copy()

# Ensure numeric features.
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")

y_raw = pd.to_numeric(df[target_col], errors="coerce")
y_binary = (y_raw == 1).astype(int)
class_names = ["Non-Seizure (y=2-5)", "Seizure (y=1)"]

print("Target mapping: seizure=1 if y==1 else non-seizure=0")
print("Raw shape:", df.shape)
print("Feature shape:", X.shape)
print("Binary class distribution:")
print(y_binary.value_counts().sort_index())

# --- CELL 2: Data Cleaning & Statistical Summary ---

print("\n--- Missing Values ---")
print(X.isnull().sum().sum(), "total missing values")
for col in X.columns:
    if X[col].isnull().any():
        X[col] = X[col].fillna(X[col].mean())

print("\n--- Descriptive Statistics (first 12 features) ---")
print(X.iloc[:, :12].describe())

print("\n--- Skewness and Kurtosis (first 20 features) ---")
print(X.skew().head(20))
print(X.kurt().head(20))

# --- CELL 3: Correlation, Covariance & ANOVA ---

print("\n--- Covariance Matrix (first 12x12 block) ---")
cov_matrix = X.cov()
print(cov_matrix.iloc[:12, :12])

print("\n--- Correlation Matrix (first 12x12 block) ---")
corr_matrix = X.corr()
print(corr_matrix.iloc[:12, :12])

plt.figure(figsize=(12, 10))
subset_cols = X.columns[:20]
sns.heatmap(corr_matrix.loc[subset_cols, subset_cols], annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap (First 20 Features)")
plt.tight_layout()
plt.show()

print("\n--- ANOVA Tests (Feature vs. Binary Target) ---")
for col in X.columns:
    g0 = X.loc[y_binary == 0, col]
    g1 = X.loc[y_binary == 1, col]
    try:
        f_stat, p_value = stats.f_oneway(g0, g1)
        print(f"ANOVA for {col}: F-stat={f_stat:.4f}, P-value={p_value:.6f}")
    except ValueError as exc:
        print(f"Could not perform ANOVA for {col}: {exc}")

# --- CELL 4: Scaling + PCA + LDA Projection ---

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"\nPCA Explained Variance (2 components): {pca.explained_variance_ratio_.sum():.4f}")

lda_proj = SKLDA(n_components=1)
X_lda = lda_proj.fit_transform(X_scaled, y_binary)
print(f"LDA projection components created: {X_lda.shape[1]}")

# --- CELL 5: Train/Test Split & Model Setup ---

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_binary.to_numpy(),
    test_size=0.2,
    random_state=42,
    stratify=y_binary,
)

models = {
    "Naive Bayesian": GaussianNB(),
    "Decision Tree (Entropy)": DecisionTreeClassifier(criterion="entropy", random_state=42),
    "K-NN (Euclidean)": KNeighborsClassifier(n_neighbors=5, p=2),
    "K-NN (Manhattan)": KNeighborsClassifier(n_neighbors=5, p=1),
    "LDA Classifier": SKLDA(),
    "PCA + K-NN": KNeighborsClassifier(n_neighbors=5),
}

results = {}
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

print("\n--- 5. MODEL IMPLEMENTATION AND EVALUATION ---")

# --- CELL 6: Model Training, Evaluation, ROC ---

for name, model in models.items():
    print(f"\n--- Model: {name} ---")

    if "PCA" in name:
        pca_step = PCA(n_components=2)
        X_train_model = pca_step.fit_transform(X_train)
        X_test_model = pca_step.transform(X_test)
        cv_scores = cross_val_score(model, X_train_model, y_train, cv=kfold, scoring="accuracy")
    elif "LDA Classifier" in name:
        lda_step = SKLDA(n_components=1)
        X_train_model = lda_step.fit_transform(X_train, y_train)
        X_test_model = lda_step.transform(X_test)
        cv_scores = cross_val_score(model, X_train_model, y_train, cv=kfold, scoring="accuracy")
    else:
        X_train_model = X_train
        X_test_model = X_test
        cv_scores = cross_val_score(model, X_train_model, y_train, cv=kfold, scoring="accuracy")

    results[name] = {"CV_Accuracy": float(cv_scores.mean())}
    print(f"K-fold Cross-Validation Avg Accuracy: {cv_scores.mean():.4f}")

    model.fit(X_train_model, y_train)
    y_pred = model.predict(X_test_model)

    acc = accuracy_score(y_test, y_pred)
    err_rate = 1 - acc
    f1 = f1_score(y_test, y_pred, average="binary", zero_division=0)
    report = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    results[name].update(
        {
            "Test_Accuracy": float(acc),
            "Error_Rate": float(err_rate),
            "Precision": float(report["weighted avg"]["precision"]),
            "Recall": float(report["weighted avg"]["recall"]),
            "F1_Score": float(f1),
        }
    )

    print(f"Test Accuracy: {acc:.4f}, Error Rate: {err_rate:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=class_names))

    cm = confusion_matrix(y_test, y_pred)
    results[name]["Confusion_Matrix"] = cm.tolist()
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix: {name}")
    plt.ylabel("Actual Class")
    plt.xlabel("Predicted Class")
    plt.tight_layout()
    plt.show()

    train_acc = model.score(X_train_model, y_train)
    print(f"Training Accuracy: {train_acc:.4f}")
    if (train_acc - acc) > 0.1:
        results[name]["Overfit_Analysis"] = "Potential Overfitting (Train Acc >> Test Acc)"
    elif acc < 0.6:
        results[name]["Overfit_Analysis"] = "Potential Underfitting (Low Acc)"
    else:
        results[name]["Overfit_Analysis"] = "Balanced (Train Acc \u2248 Test Acc)"
    print(f"Overfitting Analysis: {results[name]['Overfit_Analysis']}")

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_model)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(7, 7))
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve for {name}")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()

# --- CELL 7: Probabilistic Prediction using GaussianNB ---

print("\n--- Probabilistic Model using Gaussian Naive Bayes (Sklearn) ---")
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred_nb = gnb.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred_nb))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_nb, target_names=class_names))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_nb))

# --- FIXED SAMPLE PREDICTION ---
sample_input = np.mean(X_train, axis=0).reshape(1, -1)
print("\nSample Input (mean of scaled features):")
print(sample_input)

prob_pred = gnb.predict_proba(sample_input)
print("\nPredicted probabilities for sample input:", prob_pred)

# --- CELL 8: Final Comparison Chart ---

comparison_df = pd.DataFrame(
    {k: v for k, v in results.items() if "Test_Accuracy" in v}
).T[["Test_Accuracy", "F1_Score", "Error_Rate"]]
comparison_df = comparison_df.dropna()

print("\n--- FINAL MODEL COMPARISON TABLE ---")
print(comparison_df.sort_values(by="Test_Accuracy", ascending=False))

plt.figure(figsize=(10, 6))
comparison_df["Test_Accuracy"].sort_values().plot(kind="barh", color="skyblue")
plt.title("Comparison of Model Test Accuracies")
plt.xlabel("Test Accuracy")
plt.tight_layout()
plt.show()
