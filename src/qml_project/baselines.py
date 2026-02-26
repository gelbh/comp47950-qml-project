"""
Classical ML baselines for COMP47950 QML project.

Logistic Regression, Random Forest, and SVM. Returns accuracy, F1 (macro),
precision (macro), recall (macro), confusion matrix, and timing data for each model.
"""

import time
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.svm import SVC

_DEFAULT_RANDOM_STATE = 42

# Default models: Logistic Regression, Random Forest, SVM
DEFAULT_MODELS: dict[str, Any] = {
    "Logistic Regression": LogisticRegression(
        random_state=_DEFAULT_RANDOM_STATE,
        max_iter=1000,
    ),
    "Random Forest": RandomForestClassifier(
        random_state=_DEFAULT_RANDOM_STATE,
    ),
    "SVM": SVC(
        random_state=_DEFAULT_RANDOM_STATE,
        kernel="rbf",
    ),
}


def run_baseline(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """
    Train a model and return metrics, predictions, and timing data.

    Returns:
        dict with keys: accuracy, f1_macro, precision_macro, recall_macro,
        confusion_matrix, y_pred, n_classes, train_time_s, inference_time_s
    """
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    start = time.perf_counter()
    y_pred = model.predict(X_test)
    inference_time = time.perf_counter() - start

    n_classes = len(np.unique(y_train))
    avg = "macro" if n_classes > 2 else "binary"

    zd = "warn"
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average=avg, zero_division=zd)),
        "precision_macro": float(precision_score(
            y_test, y_pred, average=avg, zero_division=zd
        )),
        "recall_macro": float(recall_score(
            y_test, y_pred, average=avg, zero_division=zd
        )),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_pred": y_pred,
        "n_classes": n_classes,
        "train_time_s": float(train_time),
        "inference_time_s": float(inference_time),
    }


def evaluate_baselines(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    models: dict[str, Any] | None = None,
    mlflow_experiment: str | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Evaluate all baseline models on the given data.

    Parameters
    ----------
    mlflow_experiment : str or None
        If provided, log all runs to this MLflow experiment.

    Returns
    -------
    dict mapping model name -> result from run_baseline
    """
    models = models or DEFAULT_MODELS
    results = {}

    # MLflow experiment setup
    if mlflow_experiment:
        try:
            import mlflow
            mlflow.set_experiment(mlflow_experiment)
        except ImportError:
            print("Warning: MLflow not available, skipping logging")
            mlflow_experiment = None

    for name, model in models.items():
        metrics = run_baseline(model, X_train, X_test, y_train, y_test)
        results[name] = metrics

        # Log to MLflow
        if mlflow_experiment:
            try:
                import mlflow
                with mlflow.start_run(run_name=name):
                    mlflow.log_params({
                        "model_type": type(model).__name__,
                        "n_features": X_train.shape[1],
                        "n_classes": metrics["n_classes"],
                        "n_train_samples": X_train.shape[0],
                        "n_test_samples": X_test.shape[0],
                    })
                    mlflow.log_metrics({
                        "accuracy": metrics["accuracy"],
                        "f1_macro": metrics["f1_macro"],
                        "precision_macro": metrics["precision_macro"],
                        "recall_macro": metrics["recall_macro"],
                        "training_time": metrics["train_time_s"],
                        "inference_time": metrics["inference_time_s"],
                    })
            except Exception as e:
                print(f"Warning: MLflow logging failed for {name}: {e}")

    return results
