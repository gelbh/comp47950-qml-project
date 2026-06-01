"""Default QSVM sweep parameters reused across the notebook (Section 6).

Keeping these as module-level constants means MLflow cache keys stay aligned
when callers omit explicit keyword arguments to
:func:`qml_project.qsvm.tuning.run_qsvm_tuning_workflow_dataframe`.
"""

from __future__ import annotations

QSVM_ENCODINGS: tuple[str, ...] = ("angle", "amplitude", "binary")
QSVM_TRAIN_SIZES: tuple[int | str, ...] = (25, 50, 100, 150, "full")
QSVM_SEEDS: tuple[int, ...] = tuple(range(10))
QSVM_CLASS_WEIGHT: str | None = "balanced"
QSVM_M: int = 7
QSVM_COMPUTE_WIN_RATE: bool = True
QSVM_N_GAMES_WIN_RATE: int = 200

__all__ = [
    "QSVM_CLASS_WEIGHT",
    "QSVM_COMPUTE_WIN_RATE",
    "QSVM_ENCODINGS",
    "QSVM_M",
    "QSVM_N_GAMES_WIN_RATE",
    "QSVM_SEEDS",
    "QSVM_TRAIN_SIZES",
]
