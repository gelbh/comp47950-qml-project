"""Plotly / Matplotlib helpers for the Nim Streamlit demo (barrel re-exports).

Submodules group board UI, candidate charts, encoding/circuit renders, kernels,
classical sweeps, results scatters, and history bars. Pages keep ``import viz``.
"""

from __future__ import annotations

from ._constants import (
    COLOR_AGREE,
    COLOR_CHOSEN,
    COLOR_DISAGREE,
    COLOR_HEAP,
    COLOR_NEUTRAL,
    COLOR_OPTIMAL,
    COLOR_STONE_EMPTY,
    KERNEL_HEATMAP_AXIS_TICKS_MAX_N,
    KERNEL_HEATMAP_RICH_HOVER_MAX_N,
)
from .board import board_editor_figure, board_figure, move_label, nim_sum_table, state_label
from .candidates import candidate_score_bar, classical_feature_bar, stacked_class_probs
from .classical import (
    classical_baseline_models_bar,
    classical_feature_ablation_bar,
    classical_sweep_bar,
    classical_train_pca_scatter_grid,
    nim_h1_h2_winloss_heatmaps,
    vqc_qsvm_training_time_grouped_bar,
)
from .encoding import (
    cap_qiskit_mpl_figure,
    encoding_heatmap,
    render_encoding_circuit,
    render_qiskit_circuit,
)
from .history import agreement_history_figure, device_history_bar
from .kernels import kernel_matrix_heatmap, kernel_row_heatmap, sv_contributions_bar
from .results import (
    build_combined_selection_view,
    combined_selection_cost_scatter,
    selection_table_scatter,
    three_way_results_bar,
)

__all__ = [
    "COLOR_AGREE",
    "COLOR_CHOSEN",
    "COLOR_DISAGREE",
    "COLOR_HEAP",
    "COLOR_NEUTRAL",
    "COLOR_OPTIMAL",
    "COLOR_STONE_EMPTY",
    "KERNEL_HEATMAP_AXIS_TICKS_MAX_N",
    "KERNEL_HEATMAP_RICH_HOVER_MAX_N",
    "agreement_history_figure",
    "board_editor_figure",
    "board_figure",
    "build_combined_selection_view",
    "candidate_score_bar",
    "cap_qiskit_mpl_figure",
    "classical_baseline_models_bar",
    "classical_feature_ablation_bar",
    "classical_feature_bar",
    "classical_sweep_bar",
    "classical_train_pca_scatter_grid",
    "combined_selection_cost_scatter",
    "device_history_bar",
    "encoding_heatmap",
    "kernel_matrix_heatmap",
    "kernel_row_heatmap",
    "move_label",
    "nim_h1_h2_winloss_heatmaps",
    "nim_sum_table",
    "render_encoding_circuit",
    "render_qiskit_circuit",
    "selection_table_scatter",
    "stacked_class_probs",
    "state_label",
    "sv_contributions_bar",
    "three_way_results_bar",
    "vqc_qsvm_training_time_grouped_bar",
]
