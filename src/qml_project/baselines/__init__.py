"""Classical ML baselines for the Nim QML project.

This package provides feature engineering, kernel diagnostics, model
factories, sweep orchestration, and game-level win-rate evaluation used by
the classical baseline pipeline.
"""

from qml_project.baselines.evaluation import ClassicalResult, evaluate_model, run_baseline
from qml_project.baselines.ablation import (
    parity_ablation_summary_display,
    run_parity_feature_ablation_sweep,
)
from qml_project.baselines.features import (
    FEATURE_SET_DESCRIPTIONS,
    PARITY_ABLATION_FEATURE_SETS,
    FeatureSet,
    engineer_parity_features,
    prepare_features,
)
from qml_project.baselines.kernels import (
    angle_encoding_kernel,
    centered_kernel_alignment,
    compare_kernels_for_nim,
    kernel_class_separation,
    label_kernel_binary,
)
from qml_project.baselines.models import create_models
from qml_project.baselines.nim_policy import evaluate_win_rate, model_policy
from qml_project.baselines.plots import (
    CLASSICAL_CFG_LEARNING_CURVE_COLORS,
    KERNEL_ALIGNED_MODEL_COLORS,
    format_classical_sweep_summary_display,
    format_kernel_aligned_baseline_display,
    plot_classical_sample_efficiency_curves,
    plot_kernel_aligned_baseline_curves,
    plot_parity_feature_ablation_balanced_accuracy,
)
from qml_project.baselines.sweep import (
    ClassicalSweepTask,
    classical_sweep_pool_init,
    classical_sweep_worker,
    execute_classical_sweep_task,
    load_classical_sweep_cache,
    log_classical_mlflow_run,
    run_classical_sweep,
)
from qml_project.baselines.sweep_results import SweepResults

__all__ = [
    "ClassicalResult",
    "SweepResults",
    "FeatureSet",
    "FEATURE_SET_DESCRIPTIONS",
    "PARITY_ABLATION_FEATURE_SETS",
    "CLASSICAL_CFG_LEARNING_CURVE_COLORS",
    "KERNEL_ALIGNED_MODEL_COLORS",
    "format_classical_sweep_summary_display",
    "format_kernel_aligned_baseline_display",
    "parity_ablation_summary_display",
    "plot_classical_sample_efficiency_curves",
    "plot_kernel_aligned_baseline_curves",
    "plot_parity_feature_ablation_balanced_accuracy",
    "run_parity_feature_ablation_sweep",
    "prepare_features",
    "engineer_parity_features",
    "create_models",
    "evaluate_model",
    "angle_encoding_kernel",
    "centered_kernel_alignment",
    "label_kernel_binary",
    "kernel_class_separation",
    "compare_kernels_for_nim",
    "model_policy",
    "evaluate_win_rate",
    "ClassicalSweepTask",
    "classical_sweep_pool_init",
    "classical_sweep_worker",
    "execute_classical_sweep_task",
    "load_classical_sweep_cache",
    "log_classical_mlflow_run",
    "run_classical_sweep",
    "run_baseline",
]
