"""VQC noise-design sweep (depolarising, readout correction, ZNE).

Split into:

- :mod:`qml_project.training.noise_sweep.single_run` — pure per-task
  train/evaluate helper :func:`_single_noise_run`.
- :mod:`qml_project.training.noise_sweep.mlflow_io` — cache loader and
  per-run logger.
- :mod:`qml_project.training.noise_sweep.runner` — pool/worker entry points
  and the top-level :func:`run_vqc_noise_sweep`.
"""

from __future__ import annotations

from .runner import run_vqc_noise_sweep

__all__ = ["run_vqc_noise_sweep"]
