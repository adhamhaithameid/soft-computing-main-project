from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, RobustScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42

# Stage registries requested by project requirements.
PREPROCESSING_METHODS: List[str] = [
    "standard",
    "minmax",
    "robust",
    "quantile",
]

REDUCTION_METHODS: List[str] = [
    "none",
    "pca",
    "lda_projection",
    "svd",
]

SELECTION_METHODS: List[str] = [
    "none",
    "filter_chi2",
    "filter_anova",
    "filter_correlation",
    "wrapper_sfs",
    "wrapper_rfe",
    "embedded_l1",
    "ga_selection",
]

CLASSIFIER_METHODS: List[str] = [
    "knn",
    "svm",
    "decision_tree",
    "logistic_regression",
    "lda_classifier",
    "mlp_ann",
]

TRACKS: List[str] = [
    "binary",
    "multiclass",
]


@dataclass(frozen=True)
class ComboStatus:
    status: str
    skip_reason: str


@dataclass(frozen=True)
class CartesianSpec:
    preprocessing: Tuple[str, ...] = tuple(PREPROCESSING_METHODS)
    reduction: Tuple[str, ...] = tuple(REDUCTION_METHODS)
    selection: Tuple[str, ...] = tuple(SELECTION_METHODS)
    classifiers: Tuple[str, ...] = tuple(CLASSIFIER_METHODS)
    tracks: Tuple[str, ...] = tuple(TRACKS)
    cv_splits: int = 3

    @property
    def expected_combos(self) -> int:
        return (
            len(self.preprocessing)
            * len(self.reduction)
            * len(self.selection)
            * len(self.classifiers)
            * len(self.tracks)
        )

    @property
    def expected_fold_evals(self) -> int:
        return self.expected_combos * self.cv_splits


def build_preprocessor(name: str):
    if name == "standard":
        return StandardScaler()
    if name == "minmax":
        return MinMaxScaler()
    if name == "robust":
        return RobustScaler()
    if name == "quantile":
        return QuantileTransformer(output_distribution="normal", random_state=RANDOM_STATE)
    raise ValueError(f"Unknown preprocessing method: {name}")


def model_registry() -> Dict[str, object]:
    return {
        # Keep per-model threading conservative; outer cartesian parallelism controls CPU fan-out.
        "knn": KNeighborsClassifier(n_neighbors=5, n_jobs=1),
        "svm": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=RANDOM_STATE),
        "decision_tree": DecisionTreeClassifier(max_depth=12, random_state=RANDOM_STATE),
        "logistic_regression": LogisticRegression(max_iter=1000, solver="lbfgs"),
        "lda_classifier": LinearDiscriminantAnalysis(),
        "mlp_ann": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=350, random_state=RANDOM_STATE),
    }


def build_classifier(name: str):
    models = model_registry()
    if name not in models:
        raise ValueError(f"Unknown classifier: {name}")
    return models[name]


def clamp_feature_count(requested: int, max_allowed: int, minimum: int = 1) -> int:
    if max_allowed < minimum:
        return minimum
    return max(minimum, min(requested, max_allowed))


def non_negative_transform_for_chi2(
    X_train: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Chi-square requires non-negative values; shift data when needed."""
    train_min = float(np.nanmin(X_train))
    test_min = float(np.nanmin(X_test))
    floor = min(train_min, test_min)
    if floor >= 0:
        return X_train, X_test

    shift = abs(floor) + 1e-9
    return X_train + shift, X_test + shift


def skip_status(reason: str) -> ComboStatus:
    return ComboStatus(status="failed", skip_reason=reason)


def ok_status() -> ComboStatus:
    return ComboStatus(status="ok", skip_reason="")


def failure_reason(exc: Exception) -> str:
    name = exc.__class__.__name__
    text = str(exc).strip().replace("\n", " ")
    if len(text) > 200:
        text = text[:197] + "..."
    return f"{name}: {text}"
