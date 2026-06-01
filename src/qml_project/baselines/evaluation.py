"""Evaluation datatypes and metric helpers for classical baselines."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

# int 0 avoids UndefinedMetricWarning spam when predictions are single-class.
_SKLEARN_ZERO_DIVISION: int = 0


@dataclass
class ClassicalResult:
    """Full evaluation result for a single classical model run.

    When loaded from MLflow cache, ``cm`` and ``y_pred`` are None
    (not logged to MLflow). Downstream code should check before using.
    """

    model_name: str
    accuracy: float
    balanced_accuracy: float
    mcc: float
    f1: float
    precision: float
    recall: float
    cm: np.ndarray | None = None
    y_pred: np.ndarray | None = None
    train_time_s: float = 0.0
    inference_time_s: float = 0.0
    # Sweep metadata
    seed: int = 42
    train_size: int | str = "full"
    feature_set: str = "raw"
    symmetry: str = "none"
    regime: str = "ood"
    win_rate: float | None = None
    c_svc: float = 1.0


def evaluate_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    model_name: str = "",
) -> ClassicalResult:
    """Train a model and compute full evaluation metrics."""
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    inference_time = time.perf_counter() - t0

    zd = _SKLEARN_ZERO_DIVISION
    return ClassicalResult(
        model_name=model_name,
        accuracy=float(accuracy_score(y_test, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_test, y_pred)),
        mcc=float(matthews_corrcoef(y_test, y_pred)),
        f1=float(f1_score(y_test, y_pred, average="binary", zero_division=zd)),
        precision=float(
            precision_score(y_test, y_pred, average="binary", zero_division=zd)
        ),
        recall=float(recall_score(y_test, y_pred, average="binary", zero_division=zd)),
        cm=confusion_matrix(y_test, y_pred),
        y_pred=y_pred,
        train_time_s=float(train_time),
        inference_time_s=float(inference_time),
    )


def run_baseline(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Train a model and return metrics dict (legacy interface).

    Prefer :func:`evaluate_model` for new code.
    """
    result = evaluate_model(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
    )
    return {
        "accuracy": result.accuracy,
        "balanced_accuracy": result.balanced_accuracy,
        "mcc": result.mcc,
        "f1_macro": result.f1,
        "precision_macro": result.precision,
        "recall_macro": result.recall,
        "confusion_matrix": result.cm,
        "y_pred": result.y_pred,
        "n_classes": len(np.unique(y_train)),
        "train_time_s": result.train_time_s,
        "inference_time_s": result.inference_time_s,
    }
