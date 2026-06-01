"""Notebook figures mixing quantum bundles and classical aggregates (§7, §9, §11).

Split into three submodules:

- :mod:`qml_project.report_plots.styles` — pipeline style dictionaries and
  shared helpers (:func:`train_sizes_to_float_array`,
  :func:`learning_curve_mean_std_by_train_size`).
- :mod:`qml_project.report_plots.vqc_and_arch` — VQC tuning, VQC robustness,
  and architecture-diagnostics triptych (§4–§5).
- :mod:`qml_project.report_plots.comparison` — quantum selection Pareto, deep
  dive, classical overlay, and final comparison (§7, §9, §11).
"""

from __future__ import annotations

from .comparison import (
    plot_final_comparison_balanced_accuracy,
    plot_quantum_selection_pareto,
    plot_quantum_vs_classical_balanced_accuracy_overlay,
    plot_quantum_winner_two_metric_panels,
)
from .styles import (
    FINAL_COMPARISON_PIPELINE_STYLES,
    QUANTUM_WINNER_PIPELINE_STYLES,
    learning_curve_mean_std_by_train_size,
    train_sizes_to_float_array,
)
from .vqc_and_arch import (
    plot_architecture_diagnostics_triptych,
    plot_vqc_robustness_balanced_accuracy_vs_noise,
    plot_vqc_tuning_curves_by_encoding,
)

__all__ = [
    "FINAL_COMPARISON_PIPELINE_STYLES",
    "QUANTUM_WINNER_PIPELINE_STYLES",
    "learning_curve_mean_std_by_train_size",
    "plot_architecture_diagnostics_triptych",
    "plot_final_comparison_balanced_accuracy",
    "plot_quantum_selection_pareto",
    "plot_quantum_vs_classical_balanced_accuracy_overlay",
    "plot_quantum_winner_two_metric_panels",
    "plot_vqc_robustness_balanced_accuracy_vs_noise",
    "plot_vqc_tuning_curves_by_encoding",
    "train_sizes_to_float_array",
]
