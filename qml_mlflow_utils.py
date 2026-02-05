"""
Helper for consistent MLflow param/metric names in COMP47950 QML experiments.
Use from Qiskit or PennyLane runs so logging stays comparable across the module.
"""
from __future__ import annotations

import mlflow
from pathlib import Path
from typing import Any

# Standard param names for QML runs (use in log_params)
PARAM_OPTIMIZER = "optimizer"
PARAM_REPS = "reps"
PARAM_FEATURE_MAP = "feature_map"
PARAM_ANSATZ = "ansatz"
PARAM_RANDOM_STATE = "random_state"
PARAM_FRAMEWORK = "framework"  # e.g. "qiskit" or "pennylane"
PARAM_N_QUBITS = "n_qubits"
PARAM_MAX_ITER = "max_iter"

# Standard metric names
METRIC_TRAIN_ACCURACY = "train_accuracy"
METRIC_TEST_ACCURACY = "test_accuracy"
METRIC_LOSS = "loss"
METRIC_FINAL_LOSS = "final_loss"


def log_qml_run(
    params: dict[str, Any],
    metrics: dict[str, float],
    artifact_paths: list[str | Path] | None = None,
    experiment_name: str | None = None,
) -> None:
    """
    Log a QML experiment with consistent keys. Call inside mlflow.start_run().
    Optionally pass paths to log as artifacts (e.g. convergence plot, config).
    """
    if experiment_name:
        mlflow.set_experiment(experiment_name)
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    if artifact_paths:
        for p in artifact_paths:
            path = Path(p)
            if path.exists():
                mlflow.log_artifact(str(path))
