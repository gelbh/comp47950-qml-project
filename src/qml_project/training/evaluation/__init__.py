"""Training loop, circuit execution, and policy wrappers for the VQC.

Submodules:

- :mod:`qml_project.training.evaluation.circuits` — shot schedule, sampler
  execution, bitstring decoding, loss / prediction helpers.
- :mod:`qml_project.training.evaluation.policy` — :func:`vqc_policy` and
  :func:`evaluate_vqc_win_rate` for game-play evaluation.
- :mod:`qml_project.training.evaluation.training_loop` — :func:`train_classifier`
  (COBYLA optimisation with optional MLflow logging) and
  :func:`evaluate_classifier` (held-out evaluation).
"""

from __future__ import annotations

from .circuits import (
    DEFAULT_SHOT_SCHEDULE,
    evaluate_circuit,
    evaluate_circuit_outputs,
    shots_for_eval,
)
from .policy import evaluate_vqc_win_rate, vqc_policy
from .training_loop import evaluate_classifier, train_classifier

__all__ = [
    "DEFAULT_SHOT_SCHEDULE",
    "shots_for_eval",
    "evaluate_circuit",
    "evaluate_circuit_outputs",
    "vqc_policy",
    "evaluate_vqc_win_rate",
    "train_classifier",
    "evaluate_classifier",
]
