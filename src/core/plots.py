from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

from .benchmark import _apply_reduction, _apply_selection
from .cartesian_pipeline import RANDOM_STATE, build_classifier, build_preprocessor


def generate_cartesian_plots(
    metrics_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    figures_dir: Path,
    X_df: pd.DataFrame,
    y_binary: np.ndarray,
    cv_splits: int,
) -> List[Path]:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception:
        return []

    figures_dir.mkdir(parents=True, exist_ok=True)
    out: List[Path] = []

    ok = metrics_df[metrics_df["status"] == "ok"].copy()

    for track in ["binary", "multiclass"]:
        sub = summary_df[summary_df["track"] == track].copy()
        if sub.empty:
            continue

        sel_pivot = (
            sub.groupby(["model", "selection"], as_index=False)["accuracy"].mean()
            .pivot(index="model", columns="selection", values="accuracy")
        )
        plt.figure(figsize=(12, 6))
        sns.heatmap(sel_pivot, annot=True, fmt=".3f", cmap="YlGnBu")
        plt.title(f"Accuracy Heatmap: Model x Selection ({track})")
        plt.tight_layout()
        p = figures_dir / f"cartesian_heatmap_model_selection_{track}.png"
        plt.savefig(p, dpi=220)
        plt.close()
        out.append(p)

        red_pivot = (
            sub.groupby(["model", "reduction"], as_index=False)["accuracy"].mean()
            .pivot(index="model", columns="reduction", values="accuracy")
        )
        plt.figure(figsize=(10, 6))
        sns.heatmap(red_pivot, annot=True, fmt=".3f", cmap="YlOrBr")
        plt.title(f"Accuracy Heatmap: Model x Reduction ({track})")
        plt.tight_layout()
        p = figures_dir / f"cartesian_heatmap_model_reduction_{track}.png"
        plt.savefig(p, dpi=220)
        plt.close()
        out.append(p)

        top = sub.sort_values(["accuracy", "f1"], ascending=False).head(12).copy()
        top["combo"] = (
            top["preprocessing"]
            + "|"
            + top["reduction"]
            + "|"
            + top["selection"]
            + "|"
            + top["model"]
        )

        plt.figure(figsize=(14, 6))
        sns.barplot(data=top, x="combo", y="accuracy", color="#2a9d8f")
        plt.xticks(rotation=75, ha="right")
        plt.title(f"Top 12 Accuracy Combinations ({track})")
        plt.tight_layout()
        p = figures_dir / f"cartesian_top_accuracy_{track}.png"
        plt.savefig(p, dpi=220)
        plt.close()
        out.append(p)

        plt.figure(figsize=(14, 6))
        sns.barplot(data=top, x="combo", y="f1", color="#264653")
        plt.xticks(rotation=75, ha="right")
        plt.title(f"Top 12 F1 Combinations ({track})")
        plt.tight_layout()
        p = figures_dir / f"cartesian_top_f1_{track}.png"
        plt.savefig(p, dpi=220)
        plt.close()
        out.append(p)

    # Fold variance: top 5 per track
    for track in ["binary", "multiclass"]:
        sub_summary = summary_df[summary_df["track"] == track].sort_values("accuracy", ascending=False).head(5)
        if sub_summary.empty:
            continue

        frames = []
        for _, r in sub_summary.iterrows():
            m = ok[
                (ok["track"] == track)
                & (ok["preprocessing"] == r["preprocessing"])
                & (ok["reduction"] == r["reduction"])
                & (ok["selection"] == r["selection"])
                & (ok["model"] == r["model"])
            ].copy()
            if m.empty:
                continue
            m["combo"] = f"{r['model']}|{r['preprocessing']}|{r['reduction']}|{r['selection']}"
            frames.append(m)

        if frames:
            fold_df = pd.concat(frames, ignore_index=True)
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=fold_df, x="fold", y="accuracy", hue="combo", marker="o")
            plt.ylim(0, 1)
            plt.title(f"Fold-wise Accuracy Variance (Top 5, {track})")
            plt.tight_layout()
            p = figures_dir / f"cartesian_fold_variance_{track}.png"
            plt.savefig(p, dpi=220)
            plt.close()
            out.append(p)

    # ROC curves for top binary combos (re-run top 5)
    binary_top = summary_df[summary_df["track"] == "binary"].sort_values("accuracy", ascending=False).head(5)
    if not binary_top.empty:
        X = X_df.to_numpy(dtype=float)
        y = y_binary
        skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)

        plt.figure(figsize=(8, 6))
        plotted = 0
        for _, combo in binary_top.iterrows():
            y_true_all = []
            y_score_all = []
            for train_idx, test_idx in skf.split(X, y):
                X_train_raw, X_test_raw = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                try:
                    pre = build_preprocessor(str(combo["preprocessing"]))
                    Xp_train = pre.fit_transform(X_train_raw)
                    Xp_test = pre.transform(X_test_raw)
                    Xr_train, Xr_test = _apply_reduction(str(combo["reduction"]), Xp_train, y_train, Xp_test)
                    Xs_train, Xs_test = _apply_selection(str(combo["selection"]), Xr_train, y_train, Xr_test)
                    model = build_classifier(str(combo["model"]))
                    model.fit(Xs_train, y_train)
                    if hasattr(model, "predict_proba"):
                        y_score = model.predict_proba(Xs_test)[:, 1]
                    elif hasattr(model, "decision_function"):
                        y_score = model.decision_function(Xs_test)
                    else:
                        continue
                    y_true_all.extend(y_test.tolist())
                    y_score_all.extend(np.asarray(y_score).tolist())
                except Exception:
                    y_true_all = []
                    y_score_all = []
                    break

            if not y_true_all or not y_score_all:
                continue

            y_true_np = np.asarray(y_true_all)
            y_score_np = np.asarray(y_score_all)
            fpr, tpr, _ = roc_curve(y_true_np, y_score_np)
            auc = roc_auc_score(y_true_np, y_score_np)
            label = (
                f"{combo['model']}|{combo['preprocessing']}|{combo['reduction']}|"
                f"{combo['selection']} (AUC={auc:.3f})"
            )
            plt.plot(fpr, tpr, label=label)
            plotted += 1

        if plotted > 0:
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
            plt.title("ROC Curves - Top Binary Cartesian Combos")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend(fontsize=7)
            plt.tight_layout()
            p = figures_dir / "cartesian_roc_curves_binary_top.png"
            plt.savefig(p, dpi=220)
            out.append(p)
        plt.close()

    return out
