"""
Minimal MLflow logging example for COMP47950 QML.
Run from project root. Same pattern works from Qiskit or PennyLane notebooks/scripts.
"""
import sys
from pathlib import Path

# Ensure project root is on path and set tracking URI
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
from mlflow_config import set_tracking_uri
from qml_mlflow_utils import (
    PARAM_OPTIMIZER,
    PARAM_REPS,
    PARAM_RANDOM_STATE,
    METRIC_TRAIN_ACCURACY,
    METRIC_TEST_ACCURACY,
    METRIC_FINAL_LOSS,
)

def main():
    set_tracking_uri()

    with mlflow.start_run(run_name="example_qml_run"):
        # Params (example values; use your real QML config)
        mlflow.log_params({
            PARAM_OPTIMIZER: "COBYLA",
            PARAM_REPS: 1,
            PARAM_RANDOM_STATE: 42,
        })

        # Metrics (example; replace with actual train/test accuracy and loss)
        mlflow.log_metrics({
            METRIC_TRAIN_ACCURACY: 0.85,
            METRIC_TEST_ACCURACY: 0.82,
            METRIC_FINAL_LOSS: 0.31,
        })

        # Optional: save a small artifact (e.g. config or plot path)
        # mlflow.log_artifact("convergence_plot.png")

    print("Run logged to ./mlruns. Start UI with: mlflow ui (from project root)")


if __name__ == "__main__":
    main()
