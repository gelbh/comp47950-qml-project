"""QSVM tuning defaults, workflow runner, and report figures (Section 6).

Split into:

- :mod:`qml_project.qsvm.tuning.defaults` — sweep constants.
- :mod:`qml_project.qsvm.tuning.summary` — variant signatures, group/agg,
  encoding labels, and the helper used by the workflow runner.
- :mod:`qml_project.qsvm.tuning.workflow` —
  :func:`run_qsvm_tuning_workflow_dataframe` which loops variants through
  :func:`qml_project.qsvm.run_quantum_kernel_sweep`.
- :mod:`qml_project.qsvm.tuning.plots` — Section 6 plotting functions.
"""

from __future__ import annotations

from .defaults import (
    QSVM_CLASS_WEIGHT,
    QSVM_COMPUTE_WIN_RATE,
    QSVM_ENCODINGS,
    QSVM_M,
    QSVM_N_GAMES_WIN_RATE,
    QSVM_SEEDS,
    QSVM_TRAIN_SIZES,
)
from .plots import (
    plot_qsvm_faceted_metric_curves,
    plot_qsvm_full_train_balanced_accuracy_bars,
    plot_qsvm_train_and_kernel_time_curves,
)
from .summary import (
    add_qsvm_encoding_label_column,
    filter_qsvm_summary_exact_statevector,
    merge_qsvm_estimator_mode_onto_summary,
    qsvm_balanced_accuracy_mean_pivot,
    qsvm_summary_group_columns,
    qsvm_variant_signature,
    summarize_qsvm_workflow_dataframe,
)
from .workflow import run_qsvm_tuning_workflow_dataframe

__all__ = [
    "QSVM_CLASS_WEIGHT",
    "QSVM_COMPUTE_WIN_RATE",
    "QSVM_ENCODINGS",
    "QSVM_M",
    "QSVM_N_GAMES_WIN_RATE",
    "QSVM_SEEDS",
    "QSVM_TRAIN_SIZES",
    "add_qsvm_encoding_label_column",
    "filter_qsvm_summary_exact_statevector",
    "merge_qsvm_estimator_mode_onto_summary",
    "plot_qsvm_faceted_metric_curves",
    "plot_qsvm_full_train_balanced_accuracy_bars",
    "plot_qsvm_train_and_kernel_time_curves",
    "qsvm_balanced_accuracy_mean_pivot",
    "qsvm_summary_group_columns",
    "qsvm_variant_signature",
    "run_qsvm_tuning_workflow_dataframe",
    "summarize_qsvm_workflow_dataframe",
]
