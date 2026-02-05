"""
MLflow configuration for COMP47950 Quantum Machine Learning.
Import this in notebooks/scripts so tracking uses the project root ./mlruns.
Run from project root so paths resolve correctly.
"""
import os
from pathlib import Path

# Project root (directory containing this file)
PROJECT_ROOT = Path(__file__).resolve().parent
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

def set_tracking_uri(uri: str | Path | None = None) -> None:
    """Set MLflow tracking URI to project ./mlruns (or custom path)."""
    import mlflow
    target = uri if uri is not None else MLRUNS_DIR
    path = Path(target) if not isinstance(target, str) else Path(target)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{path.resolve()}")
    mlflow.set_experiment("Default")

def get_tracking_uri() -> str:
    """Return the configured tracking URI (after set_tracking_uri)."""
    import mlflow
    return mlflow.get_tracking_uri()

# Optional: set via environment so mlflow ui picks it up when run from project root
if "MLFLOW_TRACKING_URI" not in os.environ:
    os.environ["MLFLOW_TRACKING_URI"] = str(MLRUNS_DIR)
