#!/usr/bin/env python3
"""Run full-course experiments for the Epileptic Seizure Recognition project."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectFromModel, SelectKBest, SequentialFeatureSelector, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    HAS_PLOTTING = True
except Exception:
    HAS_PLOTTING = False

RANDOM_STATE = 42
CV_SPLITS = 3

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "02_data" / "raw" / "epileptic_seizure_recognition" / "epileptic_seizure_data.csv"

RESULTS_TABLES = ROOT / "05_results" / "tables"
RESULTS_METRICS = ROOT / "05_results" / "metrics"
RESULTS_FIGURES = ROOT / "05_results" / "figures"
RESULTS_FOLDS = ROOT / "05_results" / "folds"
INTERIM_DIR = ROOT / "02_data" / "interim"
PROCESSED_DIR = ROOT / "02_data" / "processed"

for p in [RESULTS_TABLES, RESULTS_METRICS, RESULTS_FIGURES, RESULTS_FOLDS, INTERIM_DIR, PROCESSED_DIR]:
    p.mkdir(parents=True, exist_ok=True)


def load_dataset() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"Raw CSV not found at {RAW_CSV}. Run: python 04_src/fetch_data.py"
        )

    df = pd.read_csv(RAW_CSV)

    unnamed = [c for c in df.columns if c.lower().startswith("unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

    target_col = "y" if "y" in df.columns else df.columns[-1]

    for col in df.columns:
        if col != target_col:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    X = df.drop(columns=[target_col]).copy()
    X = X.fillna(X.median(numeric_only=True))

    y_multi = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int).to_numpy()
    y_binary = (y_multi == 1).astype(int)

    X.to_csv(PROCESSED_DIR / "features_numeric.csv", index=False)
    pd.DataFrame({"y_multiclass": y_multi, "y_binary": y_binary}).to_csv(
        PROCESSED_DIR / "targets.csv", index=False
    )

    return X, y_multi, y_binary


def save_statistical_analysis(X: pd.DataFrame) -> None:
    desc = pd.DataFrame(
        {
            "min": X.min(),
            "max": X.max(),
            "mean": X.mean(),
            "variance": X.var(),
            "std": X.std(),
            "skewness": X.apply(lambda s: skew(s.to_numpy(), bias=False)),
            "kurtosis": X.apply(lambda s: kurtosis(s.to_numpy(), bias=False)),
        }
    )
    desc.to_csv(RESULTS_TABLES / "dataset_descriptive_stats.csv")

    cov = X.cov()
    corr = X.corr()
    cov.to_csv(RESULTS_TABLES / "covariance_matrix.csv")
    corr.to_csv(RESULTS_TABLES / "correlation_matrix.csv")

    if HAS_PLOTTING:
        plt.figure(figsize=(14, 10))
        sns.heatmap(corr, cmap="coolwarm", center=0)
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(RESULTS_FIGURES / "correlation_heatmap.png", dpi=200)
        plt.close()


def _sample_for_selection(
    X: np.ndarray, y: np.ndarray, max_samples: int = 3000
) -> Tuple[np.ndarray, np.ndarray]:
    if len(y) <= max_samples:
        return X, y

    _, X_small, _, y_small = train_test_split(
        X,
        y,
        test_size=max_samples,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return X_small, y_small


def ga_select_mask(
    X: np.ndarray,
    y: np.ndarray,
    population_size: int = 10,
    generations: int = 4,
    mutation_rate: float = 0.03,
    crossover_rate: float = 0.8,
    max_features: int = 25,
) -> np.ndarray:
    """Simple GA for feature mask optimization using logistic-regression fitness."""

    rng = np.random.default_rng(RANDOM_STATE)
    n_features = X.shape[1]

    X_small, y_small = _sample_for_selection(X, y, max_samples=900)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_small,
        y_small,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_small,
    )

    max_features = min(max_features, n_features)
    p = min(0.25, max_features / max(1, n_features))

    def fix_mask(mask: np.ndarray) -> np.ndarray:
        if mask.sum() == 0:
            idx = rng.integers(0, n_features)
            mask[idx] = 1
        if mask.sum() > max_features:
            on_idx = np.where(mask == 1)[0]
            keep = rng.choice(on_idx, size=max_features, replace=False)
            new_mask = np.zeros(n_features, dtype=int)
            new_mask[keep] = 1
            return new_mask
        return mask

    def init_individual() -> np.ndarray:
        mask = (rng.random(n_features) < p).astype(int)
        return fix_mask(mask)

    def fitness(mask: np.ndarray) -> float:
        mask = fix_mask(mask.copy())
        cols = np.where(mask == 1)[0]
        X_tr_m = X_tr[:, cols]
        X_va_m = X_va[:, cols]

        if len(np.unique(y_tr)) < 2:
            return 0.0

        model = LinearSVC(max_iter=1500, random_state=RANDOM_STATE)
        model.fit(X_tr_m, y_tr)
        pred = model.predict(X_va_m)
        acc = accuracy_score(y_va, pred)

        # Tiny penalty for very large subsets
        penalty = 0.0015 * (len(cols) / n_features)
        return float(acc - penalty)

    population = [init_individual() for _ in range(population_size)]
    scores = [fitness(ind) for ind in population]

    for _ in range(generations):
        new_population: List[np.ndarray] = []

        while len(new_population) < population_size:
            # Tournament selection
            idxs = rng.choice(population_size, size=3, replace=False)
            p1 = population[int(idxs[np.argmax([scores[i] for i in idxs])])].copy()
            idxs = rng.choice(population_size, size=3, replace=False)
            p2 = population[int(idxs[np.argmax([scores[i] for i in idxs])])].copy()

            # Crossover
            if rng.random() < crossover_rate and n_features > 2:
                point = rng.integers(1, n_features - 1)
                c1 = np.concatenate([p1[:point], p2[point:]]).astype(int)
                c2 = np.concatenate([p2[:point], p1[point:]]).astype(int)
            else:
                c1, c2 = p1.copy(), p2.copy()

            # Mutation
            for child in [c1, c2]:
                flip = rng.random(n_features) < mutation_rate
                child[flip] = 1 - child[flip]
                child = fix_mask(child)
                new_population.append(child)
                if len(new_population) >= population_size:
                    break

        population = new_population
        scores = [fitness(ind) for ind in population]

    best_idx = int(np.argmax(scores))
    return fix_mask(population[best_idx].copy()).astype(bool)


def build_feature_sets(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    feature_sets: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
        "original": (X_train, X_test)
    }

    # PCA
    pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
    feature_sets["pca"] = (pca.fit_transform(X_train), pca.transform(X_test))

    # LDA reduction (supervised)
    n_classes = len(np.unique(y_train))
    if n_classes > 1:
        n_comp = min(max(1, n_classes - 1), X_train.shape[1])
        lda_red = LinearDiscriminantAnalysis(n_components=n_comp)
        feature_sets["lda_reduction"] = (
            lda_red.fit_transform(X_train, y_train),
            lda_red.transform(X_test),
        )

    # SVD
    n_comp_svd = min(30, max(2, X_train.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_comp_svd, random_state=RANDOM_STATE)
    feature_sets["svd"] = (svd.fit_transform(X_train), svd.transform(X_test))

    # Filter selection (ANOVA)
    kbest_k = min(20, X_train.shape[1])
    kbest = SelectKBest(score_func=f_classif, k=kbest_k)
    feature_sets["filter_kbest"] = (
        kbest.fit_transform(X_train, y_train),
        kbest.transform(X_test),
    )

    # Wrapper selection (SFS) on sampled subset for speed
    X_sel, y_sel = _sample_for_selection(X_train, y_train, max_samples=700)
    n_select = min(8, X_train.shape[1])
    sfs_estimator = LinearSVC(max_iter=1500, random_state=RANDOM_STATE)
    sfs = SequentialFeatureSelector(
        sfs_estimator,
        n_features_to_select=n_select,
        direction="forward",
        scoring="accuracy",
        cv=2,
        n_jobs=1,
    )
    sfs.fit(X_sel, y_sel)
    mask_sfs = sfs.get_support()
    feature_sets["wrapper_sfs"] = (X_train[:, mask_sfs], X_test[:, mask_sfs])

    # Embedded selection (L1)
    l1 = LinearSVC(
        penalty="l1",
        dual=False,
        C=0.5,
        max_iter=2000,
        random_state=RANDOM_STATE,
    )
    embedded = SelectFromModel(l1, threshold="median")
    feature_sets["embedded_l1"] = (
        embedded.fit_transform(X_train, y_train),
        embedded.transform(X_test),
    )

    # GA selection
    ga_mask = ga_select_mask(X_train, y_train)
    feature_sets["ga_selection"] = (X_train[:, ga_mask], X_test[:, ga_mask])

    return feature_sets


def model_registry() -> Dict[str, object]:
    return {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=RANDOM_STATE),
        "DecisionTree": DecisionTreeClassifier(max_depth=12, random_state=RANDOM_STATE),
        "LogisticRegression": LogisticRegression(max_iter=1000, solver="lbfgs"),
        "LDA_Classifier": LinearDiscriminantAnalysis(),
        "MLP_ANN": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=350, random_state=RANDOM_STATE),
    }


def evaluate_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    binary: bool,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray | None]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    average = "binary" if binary else "macro"

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_test, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_test, y_pred, average=average, zero_division=0),
    }
    metrics["error_rate"] = 1.0 - metrics["accuracy"]

    y_score = None
    if binary:
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)

        if y_score is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(y_test, y_score)
            except Exception:
                metrics["roc_auc"] = np.nan
        else:
            metrics["roc_auc"] = np.nan
    else:
        metrics["roc_auc"] = np.nan

    return metrics, y_pred, y_score


def main() -> None:
    X_df, y_multi, y_binary = load_dataset()
    save_statistical_analysis(X_df)

    X = X_df.to_numpy(dtype=float)

    tracks = {
        "binary": y_binary,
        "multiclass": y_multi,
    }

    records: List[Dict[str, object]] = []
    oof_store: Dict[Tuple[str, str, str], Dict[str, List[float]]] = {}

    for track_name, y in tracks.items():
        binary = track_name == "binary"
        skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
            print(f"Running track={track_name} fold={fold}/{CV_SPLITS} ...")
            X_train_raw, X_test_raw = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train_raw)
            X_test = scaler.transform(X_test_raw)

            feature_sets = build_feature_sets(X_train, y_train, X_test)

            for feat_name, (Xf_train, Xf_test) in feature_sets.items():
                for model_name, model in model_registry().items():
                    metrics, y_pred, y_score = evaluate_model(
                        model, Xf_train, y_train, Xf_test, y_test, binary=binary
                    )

                    row = {
                        "track": track_name,
                        "fold": fold,
                        "feature_set": feat_name,
                        "model": model_name,
                        **metrics,
                    }
                    records.append(row)

                    key = (track_name, feat_name, model_name)
                    if key not in oof_store:
                        oof_store[key] = {"y_true": [], "y_pred": [], "y_score": []}

                    oof_store[key]["y_true"].extend(y_test.tolist())
                    oof_store[key]["y_pred"].extend(y_pred.tolist())
                    if y_score is not None:
                        oof_store[key]["y_score"].extend(np.asarray(y_score).tolist())

    df_metrics = pd.DataFrame(records)
    df_metrics.to_csv(RESULTS_METRICS / "metrics_all.csv", index=False)
    df_metrics.to_csv(RESULTS_FOLDS / "fold_metrics.csv", index=False)

    summary = (
        df_metrics.groupby(["track", "feature_set", "model"], as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            error_rate=("error_rate", "mean"),
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            f1=("f1", "mean"),
            roc_auc=("roc_auc", "mean"),
        )
        .sort_values(["track", "accuracy"], ascending=[True, False])
    )
    summary.to_csv(RESULTS_TABLES / "summary_accuracy.csv", index=False)

    # Plot: accuracy heatmap and fold charts per track
    if HAS_PLOTTING:
        for track in ["binary", "multiclass"]:
            sub = summary[summary["track"] == track]
            pivot = sub.pivot(index="model", columns="feature_set", values="accuracy")

            plt.figure(figsize=(12, 6))
            sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu")
            plt.title(f"Mean CV Accuracy - {track}")
            plt.tight_layout()
            plt.savefig(RESULTS_FIGURES / f"accuracy_heatmap_{track}.png", dpi=220)
            plt.close()

            # Fold-wise line chart on original feature set
            sub_fold = df_metrics[(df_metrics["track"] == track) & (df_metrics["feature_set"] == "original")]
            plt.figure(figsize=(10, 5))
            sns.lineplot(data=sub_fold, x="fold", y="accuracy", hue="model", marker="o")
            plt.title(f"Fold-wise Accuracy (Original Features) - {track}")
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(RESULTS_FIGURES / f"fold_accuracy_original_{track}.png", dpi=220)
            plt.close()

    # Confusion matrices for best model per track
    for track in ["binary", "multiclass"]:
        best = summary[summary["track"] == track].iloc[0]
        key = (track, best["feature_set"], best["model"])

        y_true = np.array(oof_store[key]["y_true"])
        y_pred = np.array(oof_store[key]["y_pred"])

        cm = confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(cm)
        cm_df.to_csv(
            RESULTS_TABLES / f"confusion_matrix_best_{track}.csv",
            index=False,
        )

        if HAS_PLOTTING:
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title(f"Confusion Matrix (Best: {best['model']} + {best['feature_set']}) - {track}")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.tight_layout()
            plt.savefig(RESULTS_FIGURES / f"confusion_matrix_best_{track}.png", dpi=220)
            plt.close()

    # ROC curves for top binary models with scores
    if HAS_PLOTTING:
        binary_best = summary[summary["track"] == "binary"].head(6)
        plt.figure(figsize=(8, 6))
        plotted = 0
        for _, row in binary_best.iterrows():
            key = ("binary", row["feature_set"], row["model"])
            y_true = np.array(oof_store[key]["y_true"])
            y_score = np.array(oof_store[key]["y_score"])
            if y_score.size == 0:
                continue

            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc = roc_auc_score(y_true, y_score)
            label = f"{row['model']} + {row['feature_set']} (AUC={auc:.3f})"
            plt.plot(fpr, tpr, label=label)
            plotted += 1

        if plotted > 0:
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
            plt.title("ROC Curves - Top Binary Models")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(RESULTS_FIGURES / "roc_curves_binary_top.png", dpi=220)
        plt.close()

    # Save run summary JSON
    def _json_safe_row(row_dict: Dict[str, object]) -> Dict[str, object]:
        safe: Dict[str, object] = {}
        for k, v in row_dict.items():
            if pd.isna(v):
                safe[k] = None
            elif isinstance(v, (np.integer, np.floating)):
                safe[k] = float(v)
            else:
                safe[k] = v
        return safe

    run_summary = {
        "rows": int(len(X_df)),
        "features": int(X_df.shape[1]),
        "tracks": ["binary", "multiclass"],
        "total_experiments": int(len(df_metrics)),
        "best_binary": _json_safe_row(summary[summary["track"] == "binary"].iloc[0].to_dict()),
        "best_multiclass": _json_safe_row(summary[summary["track"] == "multiclass"].iloc[0].to_dict()),
    }
    with (RESULTS_METRICS / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    print("Experiment pipeline finished.")
    print(f"Saved metrics: {RESULTS_METRICS / 'metrics_all.csv'}")
    print(f"Saved summary: {RESULTS_TABLES / 'summary_accuracy.csv'}")


if __name__ == "__main__":
    main()
