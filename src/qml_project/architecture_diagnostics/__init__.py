"""§4.3 architecture diagnostics package: orchestration + MLflow cache.

Split into:

- :mod:`qml_project.architecture_diagnostics.run` — top-level orchestrator
  :func:`run_architecture_diagnostics_dataframes` and the
  :func:`pick_safe_diagnostic_n_features` helper.
- :mod:`qml_project.architecture_diagnostics.keys` — param / metric key
  conventions (part of the cache contract).
- :mod:`qml_project.architecture_diagnostics.load` —
  :func:`load_architecture_diagnostics_cache`.
- :mod:`qml_project.architecture_diagnostics.log` —
  :func:`log_architecture_diagnostics_to_mlflow`.
"""

from __future__ import annotations

from .keys import PIPELINE, TASK_EXPRESS, TASK_GRAD
from .load import load_architecture_diagnostics_cache
from .log import log_architecture_diagnostics_to_mlflow
from .run import (
    ARCH_DIAGNOSTIC_ANSATZE,
    ARCH_DIAGNOSTIC_BASE_DEPTH,
    ARCH_DIAGNOSTIC_DEPTH_LADDER,
    pick_safe_diagnostic_n_features,
    run_architecture_diagnostics_dataframes,
)

__all__ = [
    "ARCH_DIAGNOSTIC_ANSATZE",
    "ARCH_DIAGNOSTIC_BASE_DEPTH",
    "ARCH_DIAGNOSTIC_DEPTH_LADDER",
    "PIPELINE",
    "TASK_EXPRESS",
    "TASK_GRAD",
    "load_architecture_diagnostics_cache",
    "log_architecture_diagnostics_to_mlflow",
    "pick_safe_diagnostic_n_features",
    "run_architecture_diagnostics_dataframes",
]
