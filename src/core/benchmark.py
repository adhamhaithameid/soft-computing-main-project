from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, SequentialFeatureSelector, chi2, f_classif
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import LinearSVC

from .cartesian_pipeline import (
    RANDOM_STATE,
    CartesianSpec,
    build_classifier,
    build_preprocessor,
    clamp_feature_count,
    failure_reason,
    non_negative_transform_for_chi2,
)
from .runner import RunnerIO, append_checkpoint, write_manifest


def _sample_for_selection(
    X: np.ndarray, y: np.ndarray, max_samples: int = 1200
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


def _ga_select_mask(
    X: np.ndarray,
    y: np.ndarray,
    population_size: int = 10,
    generations: int = 4,
    mutation_rate: float = 0.03,
    crossover_rate: float = 0.8,
    max_features: int = 25,
) -> np.ndarray:
    rng = np.random.default_rng(RANDOM_STATE)
    n_features = X.shape[1]
    max_features = min(max_features, n_features)

    X_small, y_small = _sample_for_selection(X, y, max_samples=900)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_small,
        y_small,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_small,
    )

    p = min(0.25, max_features / max(1, n_features))

    def fix_mask(mask: np.ndarray) -> np.ndarray:
        if mask.sum() == 0:
            mask[rng.integers(0, n_features)] = 1
        if mask.sum() > max_features:
            on_idx = np.where(mask == 1)[0]
            keep = rng.choice(on_idx, size=max_features, replace=False)
            new_mask = np.zeros(n_features, dtype=int)
            new_mask[keep] = 1
            return new_mask
        return mask

    def init_individual() -> np.ndarray:
        return fix_mask((rng.random(n_features) < p).astype(int))

    def fitness(mask: np.ndarray) -> float:
        cols = np.where(fix_mask(mask.copy()) == 1)[0]
        if cols.size == 0:
            return 0.0
        clf = LinearSVC(max_iter=5000, tol=1e-3, random_state=RANDOM_STATE)
        clf.fit(X_tr[:, cols], y_tr)
        pred = clf.predict(X_va[:, cols])
        acc = accuracy_score(y_va, pred)
        penalty = 0.0015 * (len(cols) / max(1, n_features))
        return float(acc - penalty)

    population = [init_individual() for _ in range(population_size)]
    scores = [fitness(ind) for ind in population]

    for _ in range(generations):
        new_population: List[np.ndarray] = []
        while len(new_population) < population_size:
            t1 = rng.choice(population_size, size=3, replace=False)
            p1 = population[int(t1[np.argmax([scores[i] for i in t1])])].copy()
            t2 = rng.choice(population_size, size=3, replace=False)
            p2 = population[int(t2[np.argmax([scores[i] for i in t2])])].copy()

            if rng.random() < crossover_rate and n_features > 2:
                point = rng.integers(1, n_features - 1)
                c1 = np.concatenate([p1[:point], p2[point:]]).astype(int)
                c2 = np.concatenate([p2[:point], p1[point:]]).astype(int)
            else:
                c1, c2 = p1.copy(), p2.copy()

            for child in [c1, c2]:
                flip = rng.random(n_features) < mutation_rate
                child[flip] = 1 - child[flip]
                new_population.append(fix_mask(child))
                if len(new_population) >= population_size:
                    break

        population = new_population
        scores = [fitness(ind) for ind in population]

    return population[int(np.argmax(scores))].astype(bool)


def _apply_reduction(
    method: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    n_features = X_train.shape[1]

    if method == "none":
        return X_train, X_test
    if method == "pca":
        pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
        return pca.fit_transform(X_train), pca.transform(X_test)
    if method == "lda_projection":
        n_classes = len(np.unique(y_train))
        n_comp = clamp_feature_count(n_classes - 1, n_features)
        lda = LinearDiscriminantAnalysis(n_components=n_comp)
        return lda.fit_transform(X_train, y_train), lda.transform(X_test)
    if method == "svd":
        n_comp = min(30, max(2, n_features - 1))
        svd = TruncatedSVD(n_components=n_comp, random_state=RANDOM_STATE)
        return svd.fit_transform(X_train), svd.transform(X_test)
    raise ValueError(f"Unknown reduction method: {method}")


def _apply_selection(
    method: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    selection_jobs: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    n_features = X_train.shape[1]
    if method == "none":
        return X_train, X_test

    if method == "filter_chi2":
        X_chi_train, X_chi_test = non_negative_transform_for_chi2(X_train, X_test)
        k = clamp_feature_count(20, n_features)
        fs = SelectKBest(score_func=chi2, k=k)
        return fs.fit_transform(X_chi_train, y_train), fs.transform(X_chi_test)

    if method == "filter_anova":
        k = clamp_feature_count(20, n_features)
        fs = SelectKBest(score_func=f_classif, k=k)
        return fs.fit_transform(X_train, y_train), fs.transform(X_test)

    if method == "filter_correlation":
        k = clamp_feature_count(20, n_features)
        y_num = y_train.astype(float)
        corr = []
        for idx in range(n_features):
            col = X_train[:, idx]
            c = np.corrcoef(col, y_num)[0, 1]
            if np.isnan(c):
                c = 0.0
            corr.append(abs(float(c)))
        top_idx = np.argsort(corr)[::-1][:k]
        return X_train[:, top_idx], X_test[:, top_idx]

    if method == "wrapper_sfs":
        X_sel, y_sel = _sample_for_selection(X_train, y_train, max_samples=700)
        n_select = clamp_feature_count(8, X_sel.shape[1])
        est = LinearSVC(max_iter=5000, tol=1e-3, random_state=RANDOM_STATE)
        sfs = SequentialFeatureSelector(
            est,
            n_features_to_select=n_select,
            direction="forward",
            scoring="accuracy",
            cv=2,
            n_jobs=max(1, selection_jobs),
        )
        sfs.fit(X_sel, y_sel)
        mask = sfs.get_support()
        return X_train[:, mask], X_test[:, mask]

    if method == "wrapper_rfe":
        n_select = clamp_feature_count(20, n_features)
        est = LinearSVC(max_iter=5000, tol=1e-3, random_state=RANDOM_STATE)
        rfe = RFE(estimator=est, n_features_to_select=n_select, step=0.1)
        rfe.fit(X_train, y_train)
        mask = rfe.get_support()
        return X_train[:, mask], X_test[:, mask]

    if method == "embedded_l1":
        est = LinearSVC(
            penalty="l1",
            dual=False,
            C=0.5,
            max_iter=6000,
            tol=1e-3,
            random_state=RANDOM_STATE,
        )
        sfm = SelectFromModel(estimator=est, threshold="median")
        return sfm.fit_transform(X_train, y_train), sfm.transform(X_test)

    if method == "ga_selection":
        mask = _ga_select_mask(X_train, y_train)
        return X_train[:, mask], X_test[:, mask]

    raise ValueError(f"Unknown selection method: {method}")


def _evaluate(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    binary: bool,
) -> Dict[str, float]:
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    t1 = time.perf_counter()
    y_pred = model.predict(X_test)
    t2 = time.perf_counter()

    average = "binary" if binary else "macro"
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, average=average, zero_division=0)),
    }
    metrics["error_rate"] = float(1.0 - metrics["accuracy"])

    if binary:
        y_score = None
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
        if y_score is not None:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_test, y_score))
            except Exception:
                metrics["roc_auc"] = np.nan
        else:
            metrics["roc_auc"] = np.nan
    else:
        metrics["roc_auc"] = np.nan

    metrics["fit_time_sec"] = float(t1 - t0)
    metrics["predict_time_sec"] = float(t2 - t1)
    return metrics


def _failed_metrics_row(reason: str) -> Dict[str, object]:
    return {
        "accuracy": np.nan,
        "precision": np.nan,
        "recall": np.nan,
        "f1": np.nan,
        "roc_auc": np.nan,
        "error_rate": np.nan,
        "fit_time_sec": np.nan,
        "predict_time_sec": np.nan,
        "status": "failed",
        "skip_reason": reason,
    }


def _format_hms(seconds: float) -> str:
    """Format elapsed seconds as HH:MM:SS for concise progress logs."""
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _evaluate_model_row(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    binary: bool,
) -> Dict[str, object]:
    row = {"model": model_name}
    try:
        model = build_classifier(model_name)
        metrics = _evaluate(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            binary=binary,
        )
        row.update(metrics)
        row["status"] = "ok"
        row["skip_reason"] = ""
    except Exception as exc:
        row.update(_failed_metrics_row(f"model_failed: {failure_reason(exc)}"))
    return row


def run_cartesian_benchmark(
    X_df: pd.DataFrame,
    y_binary: np.ndarray,
    y_multiclass: np.ndarray,
    spec: CartesianSpec,
    io: RunnerIO,
    resume: bool = True,
    max_rows: int | None = None,
    model_jobs: int = 1,
    selection_jobs: int = 1,
    execution_device: str = "cpu",
    acceleration_backend: str = "none",
    checkpoint_percent: int = 5,
    run_label: str = "",
    platform_profile: str = "",
) -> pd.DataFrame:
    started_perf = time.perf_counter()
    started_utc = datetime.now(timezone.utc)

    # Normalize user inputs for robust, deterministic checkpoint behavior.
    cp_percent = max(1, min(100, int(checkpoint_percent or 5)))
    target_total_rows = spec.expected_fold_evals
    if max_rows is not None:
        target_total_rows = min(spec.expected_fold_evals, max(0, int(max_rows)))

    records_buffer: List[Dict[str, object]] = []
    written_new_rows = 0

    previous_manifest: Dict[str, object] = {}
    if resume and io.manifest_json.exists():
        try:
            previous_manifest = json.loads(io.manifest_json.read_text(encoding="utf-8"))
        except Exception:
            previous_manifest = {}

    seen = set()
    existing_count = 0
    if resume and io.metrics_csv.exists():
        existing = pd.read_csv(io.metrics_csv)
        existing_count = int(len(existing))
        for _, r in existing.iterrows():
            key = (
                str(r["track"]),
                int(r["fold"]),
                str(r["preprocessing"]),
                str(r["reduction"]),
                str(r["selection"]),
                str(r["model"]),
            )
            seen.add(key)

    def total_rows_so_far() -> int:
        return existing_count + written_new_rows + len(records_buffer)

    def at_target_limit() -> bool:
        return total_rows_so_far() >= target_total_rows

    def flush_buffer(force: bool = False) -> None:
        nonlocal records_buffer, written_new_rows
        if not records_buffer:
            return
        if not force and len(records_buffer) < io.checkpoint_every:
            return

        df_chk = pd.DataFrame(records_buffer)
        overwrite = (not io.metrics_csv.exists()) and existing_count == 0 and written_new_rows == 0
        append_checkpoint(io, df_chk, overwrite=overwrite)
        written_new_rows += len(records_buffer)
        records_buffer = []

    progress_marks: List[Tuple[int, int]] = []
    for percent in range(cp_percent, 101, cp_percent):
        threshold_rows = int(np.ceil(target_total_rows * (percent / 100.0)))
        progress_marks.append((percent, max(1, threshold_rows)))

    progress_idx = 0
    while progress_idx < len(progress_marks) and total_rows_so_far() >= progress_marks[progress_idx][1]:
        progress_idx += 1

    def emit_progress_if_needed() -> None:
        nonlocal progress_idx
        current_total = total_rows_so_far()
        while progress_idx < len(progress_marks) and current_total >= progress_marks[progress_idx][1]:
            # Write a durable checkpoint exactly at each progress threshold.
            flush_buffer(force=True)
            current_total = total_rows_so_far()
            percent, _ = progress_marks[progress_idx]
            elapsed = time.perf_counter() - started_perf
            processed_this_invocation = max(1, written_new_rows + len(records_buffer))
            rate = processed_this_invocation / max(elapsed, 1e-9)
            remaining = max(0, target_total_rows - current_total)
            eta_sec = remaining / rate if rate > 0 else float("inf")
            eta_text = _format_hms(eta_sec) if np.isfinite(eta_sec) else "unknown"
            print(
                f"[checkpoint] {percent}% complete "
                f"({current_total}/{target_total_rows}) "
                f"elapsed={_format_hms(elapsed)} eta={eta_text}"
            )
            progress_idx += 1

    def add_record(row: Dict[str, object]) -> None:
        records_buffer.append(row)
        flush_buffer(force=False)
        emit_progress_if_needed()

    X = X_df.to_numpy(dtype=float)

    tracks = {
        "binary": y_binary,
        "multiclass": y_multiclass,
    }

    if target_total_rows <= 0:
        raise ValueError("target_total_rows resolved to 0. Increase --max-rows or remove it.")

    for track_name in spec.tracks:
        y = tracks[track_name]
        binary = track_name == "binary"
        skf = StratifiedKFold(n_splits=spec.cv_splits, shuffle=True, random_state=RANDOM_STATE)

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
            if at_target_limit():
                break
            X_train_raw, X_test_raw = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            for prep_name in spec.preprocessing:
                if at_target_limit():
                    break
                # Fast resume: skip expensive preprocessing branch if every downstream
                # combination for this (track, fold, preprocessing) is already present.
                pending_prep = False
                for red_name_probe in spec.reduction:
                    for sel_name_probe in spec.selection:
                        for model_name_probe in spec.classifiers:
                            key_probe = (
                                track_name,
                                fold,
                                prep_name,
                                red_name_probe,
                                sel_name_probe,
                                model_name_probe,
                            )
                            if key_probe not in seen:
                                pending_prep = True
                                break
                        if pending_prep:
                            break
                    if pending_prep:
                        break
                if not pending_prep:
                    continue

                try:
                    prep = build_preprocessor(prep_name)
                    Xp_train = prep.fit_transform(X_train_raw)
                    Xp_test = prep.transform(X_test_raw)
                except Exception as exc:
                    # If preprocessing fails, log a failed row for all downstream combinations.
                    for red_name in spec.reduction:
                        for sel_name in spec.selection:
                            for model_name in spec.classifiers:
                                if at_target_limit():
                                    break
                                key = (track_name, fold, prep_name, red_name, sel_name, model_name)
                                if key in seen:
                                    continue
                                add_record(
                                    {
                                        "track": track_name,
                                        "fold": fold,
                                        "preprocessing": prep_name,
                                        "reduction": red_name,
                                        "selection": sel_name,
                                        "model": model_name,
                                        "accuracy": np.nan,
                                        "precision": np.nan,
                                        "recall": np.nan,
                                        "f1": np.nan,
                                        "roc_auc": np.nan,
                                        "error_rate": np.nan,
                                        "fit_time_sec": np.nan,
                                        "predict_time_sec": np.nan,
                                        "status": "failed",
                                        "skip_reason": f"preprocessing_failed: {failure_reason(exc)}",
                                    }
                                )
                    continue

                for red_name in spec.reduction:
                    if at_target_limit():
                        break
                    pending_red = False
                    for sel_name_probe in spec.selection:
                        for model_name_probe in spec.classifiers:
                            key_probe = (
                                track_name,
                                fold,
                                prep_name,
                                red_name,
                                sel_name_probe,
                                model_name_probe,
                            )
                            if key_probe not in seen:
                                pending_red = True
                                break
                        if pending_red:
                            break
                    if not pending_red:
                        continue

                    reduction_ok = True
                    red_reason = ""
                    try:
                        Xr_train, Xr_test = _apply_reduction(red_name, Xp_train, y_train, Xp_test)
                    except Exception as exc:
                        reduction_ok = False
                        red_reason = f"reduction_failed: {failure_reason(exc)}"
                        Xr_train, Xr_test = Xp_train, Xp_test

                    for sel_name in spec.selection:
                        if at_target_limit():
                            break
                        model_names: List[str] = []
                        for model_name in spec.classifiers:
                            key = (track_name, fold, prep_name, red_name, sel_name, model_name)
                            if key in seen:
                                continue
                            model_names.append(model_name)
                        if not model_names:
                            continue

                        selection_ok = True
                        sel_reason = ""
                        try:
                            if reduction_ok:
                                Xs_train, Xs_test = _apply_selection(
                                    sel_name,
                                    Xr_train,
                                    y_train,
                                    Xr_test,
                                    selection_jobs=selection_jobs,
                                )
                            else:
                                selection_ok = False
                                sel_reason = red_reason
                                Xs_train, Xs_test = Xr_train, Xr_test
                        except Exception as exc:
                            selection_ok = False
                            sel_reason = f"selection_failed: {failure_reason(exc)}"
                            Xs_train, Xs_test = Xr_train, Xr_test

                        remaining = target_total_rows - total_rows_so_far()
                        if remaining <= 0:
                            break
                        model_names = model_names[:remaining]

                        base_row = {
                            "track": track_name,
                            "fold": fold,
                            "preprocessing": prep_name,
                            "reduction": red_name,
                            "selection": sel_name,
                        }

                        if not reduction_ok or not selection_ok:
                            for model_name in model_names:
                                row = dict(base_row)
                                row["model"] = model_name
                                row.update(_failed_metrics_row(sel_reason))
                                add_record(row)
                        else:
                            if model_jobs <= 1 or len(model_names) <= 1:
                                for model_name in model_names:
                                    row = dict(base_row)
                                    row.update(
                                        _evaluate_model_row(
                                            model_name=model_name,
                                            X_train=Xs_train,
                                            y_train=y_train,
                                            X_test=Xs_test,
                                            y_test=y_test,
                                            binary=binary,
                                        )
                                    )
                                    add_record(row)
                            else:
                                result_map: Dict[str, Dict[str, object]] = {}
                                with ThreadPoolExecutor(max_workers=min(model_jobs, len(model_names))) as ex:
                                    futures = {
                                        ex.submit(
                                            _evaluate_model_row,
                                            model_name,
                                            Xs_train,
                                            y_train,
                                            Xs_test,
                                            y_test,
                                            binary,
                                        ): model_name
                                        for model_name in model_names
                                    }
                                    for fut in as_completed(futures):
                                        model_name = futures[fut]
                                        try:
                                            result_map[model_name] = fut.result()
                                        except Exception as exc:
                                            result_map[model_name] = {
                                                "model": model_name,
                                                **_failed_metrics_row(
                                                    f"model_failed: {failure_reason(exc)}"
                                                ),
                                            }

                                for model_name in model_names:
                                    row = dict(base_row)
                                    row.update(result_map[model_name])
                                    add_record(row)

    # Final durable checkpoint.
    flush_buffer(force=True)
    emit_progress_if_needed()

    if not io.metrics_csv.exists():
        raise FileNotFoundError(
            f"Metrics file was not generated at {io.metrics_csv}. "
            "Check dataset availability and runtime logs."
        )

    df = pd.read_csv(io.metrics_csv)

    ok = df[df["status"] == "ok"].copy()
    summary = (
        ok.groupby(["track", "preprocessing", "reduction", "selection", "model"], as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            f1=("f1", "mean"),
            roc_auc=("roc_auc", "mean"),
        )
        .sort_values(["track", "accuracy"], ascending=[True, False])
    )

    best_binary = None
    best_multiclass = None
    if len(summary[summary["track"] == "binary"]) > 0:
        best_binary = summary[summary["track"] == "binary"].iloc[0].to_dict()
    if len(summary[summary["track"] == "multiclass"]) > 0:
        best_multiclass = summary[summary["track"] == "multiclass"].iloc[0].to_dict()

    finished_utc = datetime.now(timezone.utc)
    runtime_last_invocation = float(time.perf_counter() - started_perf)
    runtime_effective = runtime_last_invocation
    if written_new_rows == 0 and "runtime_sec" in previous_manifest:
        try:
            runtime_effective = float(previous_manifest["runtime_sec"])
        except Exception:
            runtime_effective = runtime_last_invocation

    manifest = {
        "expected_combos": spec.expected_combos,
        "expected_fold_evals": spec.expected_fold_evals,
        "target_total_rows": int(target_total_rows),
        "rows_written": int(len(df)),
        "completed_ok": int((df["status"] == "ok").sum()),
        "skipped_or_failed": int((df["status"] != "ok").sum()),
        "runtime_sec": runtime_effective,
        "runtime_sec_last_invocation": runtime_last_invocation,
        "started_utc": started_utc.isoformat(),
        "finished_utc": finished_utc.isoformat(),
        "checkpoint_percent": cp_percent,
        "resume_mode": bool(resume),
        "resume_noop": bool(written_new_rows == 0),
        "max_rows": None if max_rows is None else int(max_rows),
        "run_label": run_label,
        "platform_profile": platform_profile,
        "execution_device": execution_device,
        "acceleration_backend": acceleration_backend,
        "best_binary": best_binary,
        "best_multiclass": best_multiclass,
    }
    write_manifest(io, manifest)

    return df
