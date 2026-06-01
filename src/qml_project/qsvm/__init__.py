r"""Quantum-kernel SVM pipeline for Nim sample-efficiency experiments.

Implements the first quantum milestone from the project plan:

- Build quantum feature-map states for Nim encodings (angle, amplitude, binary).
- Compute kernel matrices
  ``k(x, x') = |<0|U^\dagger(x) U(x')|0>|^2 = |<psi(x)|psi(x')>|^2``.
- Train sklearn SVC with precomputed kernels.
- Run OOD sample-efficiency sweeps across training sizes and seeds.

Split into focused submodules:

- :mod:`qml_project.qsvm.kernel` — feature-map statevectors,
  :func:`quantum_kernel_matrix`, and validation.
- :mod:`qml_project.qsvm.model` — :class:`QuantumKernelResult`,
  :class:`QuantumKernelSweepResults`, :class:`QuantumKernelSVMModel`, plus
  fit / evaluate / policy / win-rate helpers.
- :mod:`qml_project.qsvm.mlflow_io` — cache loader, per-run logger, and the
  :data:`QSVM_ENCODING_CACHE_REVISION` constant.
- :mod:`qml_project.qsvm.sweep` — :func:`run_quantum_kernel_sweep` and the
  pickle-safe pool/worker entry points.
- :mod:`qml_project.qsvm.comparison` — :func:`build_kernel_pipeline_comparison`.
"""

from __future__ import annotations

from .comparison import build_kernel_pipeline_comparison
from .kernel import KernelBackend, KernelEstimatorMode, quantum_kernel_matrix
from .mlflow_io import QSVM_ENCODING_CACHE_REVISION
from .model import (
    QuantumKernelResult,
    QuantumKernelSVMModel,
    QuantumKernelSweepResults,
    angle_features_for_vqc,
    evaluate_qsvm_win_rate,
    evaluate_quantum_kernel_svm,
    fit_quantum_kernel_svm,
    qsvm_policy,
)
from .sweep import QsvmSweepTask, execute_qsvm_sweep_task, run_quantum_kernel_sweep

__all__ = [
    "KernelBackend",
    "KernelEstimatorMode",
    "QSVM_ENCODING_CACHE_REVISION",
    "QsvmSweepTask",
    "QuantumKernelResult",
    "QuantumKernelSVMModel",
    "QuantumKernelSweepResults",
    "angle_features_for_vqc",
    "build_kernel_pipeline_comparison",
    "evaluate_qsvm_win_rate",
    "evaluate_quantum_kernel_svm",
    "execute_qsvm_sweep_task",
    "fit_quantum_kernel_svm",
    "qsvm_policy",
    "quantum_kernel_matrix",
    "run_quantum_kernel_sweep",
]
