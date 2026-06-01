"""Classical baseline sweep primitives.

Pickle-safe task execution, MLflow cache/resume, and the
:func:`run_classical_sweep` orchestrator are split into three submodules:

- :mod:`qml_project.baselines.sweep.tasks` — :class:`ClassicalSweepTask` and the
  pool init/worker helpers (top-level for ``spawn`` pickling).
- :mod:`qml_project.baselines.sweep.cache` —
  :func:`load_classical_sweep_cache` and :func:`log_classical_mlflow_run`.
- :mod:`qml_project.baselines.sweep.runner` — :func:`run_classical_sweep` grid
  enumeration and serial/parallel dispatch.
"""

from __future__ import annotations

from .cache import load_classical_sweep_cache, log_classical_mlflow_run
from .runner import run_classical_sweep
from .tasks import (
    ClassicalSweepTask,
    classical_sweep_pool_init,
    classical_sweep_worker,
    execute_classical_sweep_task,
)

__all__ = [
    "ClassicalSweepTask",
    "classical_sweep_pool_init",
    "classical_sweep_worker",
    "execute_classical_sweep_task",
    "load_classical_sweep_cache",
    "log_classical_mlflow_run",
    "run_classical_sweep",
]
