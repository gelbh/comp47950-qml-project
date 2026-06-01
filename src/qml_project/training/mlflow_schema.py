"""Canonical MLflow param / metric / tag **string keys** and ``params.pipeline`` values.

Every symbol here is part of the **cache contract**: resume loaders filter on these
keys; loggers must write the same names. Phase 5 migration rewrites historical
runs when keys or values change — loaders assume post-migration shape only.

Namespace layout:

- :class:`ParamKey` — flat param names passed to ``mlflow.log_params``.
- :class:`PipelineValue` — allowed ``params.pipeline`` values per sweep family.
- :class:`RegimeValue` — ``params.regime`` where used (e.g. simulated VQC OOD).
- Architecture diagnostics — pipeline value + ``task`` values live here; formatting
  helpers stay in :mod:`qml_project.architecture_diagnostics.keys`.
- Filters — pre-built ``search_runs`` ``filter_string`` fragments for FINISHED runs.
"""

from __future__ import annotations


class ParamKey:
    """Logged MLflow parameter keys (string column names in the UI)."""

    PIPELINE = "pipeline"
    REGIME = "regime"
    MODEL = "model"
    FEATURE_SET = "feature_set"
    SYMMETRY = "symmetry"
    TRAIN_SIZE = "train_size"
    SEED = "seed"
    C_SVC = "c_svc"
    ENCODING = "encoding"
    INCLUDE_NIM_SUM = "include_nim_sum"
    BITS_PER_HEAP = "bits_per_heap"
    IQP_REPS = "iqp_reps"
    ESTIMATOR_MODE = "estimator_mode"
    KERNEL_BACKEND = "kernel_backend"
    SHOTS = "shots"
    VARIANT_ID = "variant_id"
    ENCODING_CACHE_REVISION = "encoding_cache_revision"
    MLFLOW_RUN_PREFIX_STAGE = "mlflow_run_prefix_stage"
    MAX_ITER = "max_iter"
    TEST_SHOTS = "test_shots"
    ANSATZ = "ansatz"
    N_QUBITS = "n_qubits"
    N_FEATURES = "n_features"
    N_TRAINABLE = "n_trainable"
    OBSERVABLE = "observable"
    DECISION_RULE = "decision_rule"
    LOSS_NAME = "loss_name"
    EXPECTATION_QUBIT = "expectation_qubit"
    N_GAMES_WIN_RATE = "n_games_win_rate"
    CONFIG_ID = "config_id"
    TASK = "task"
    RUN_PREFIX = "run_prefix"
    NOISE_PROFILE = "noise_profile"
    NOISE_LEVEL = "noise_level"


class MetricKey:
    """Common MLflow metric keys referenced by cache loaders."""

    ACCURACY = "accuracy"
    BALANCED_ACCURACY = "balanced_accuracy"
    MCC = "mcc"
    F1 = "f1"
    PRECISION = "precision"
    RECALL = "recall"
    TRAIN_TIME_S = "train_time_s"
    INFERENCE_TIME_S = "inference_time_s"
    KERNEL_MATRIX_TIME_S = "kernel_matrix_time_s"
    WIN_RATE = "win_rate"
    TRAINING_TIME = "training_time"
    INFERENCE_TIME = "inference_time"
    FINAL_LOSS = "final_loss"
    TEST_ACCURACY = "test_accuracy"


class PipelineValue:
    """Values stored under ``params.pipeline``."""

    CLASSICAL = "classical"
    QSVM = "qsvm"
    SIMULATED_VQC = "simulated_vqc"
    SIMULATED_VQC_NOISE = "simulated_vqc_noise"
    ARCHITECTURE_DIAGNOSTICS = "architecture_diagnostics"


class RegimeValue:
    """Values stored under ``params.regime`` where applicable."""

    OOD = "ood"


# --- Architecture diagnostics (also re-exported from architecture_diagnostics.keys) ---
PIPELINE_ARCHITECTURE_DIAGNOSTICS = PipelineValue.ARCHITECTURE_DIAGNOSTICS
TASK_EXPRESSIBILITY_BATCH = "expressibility_batch"
TASK_GRADIENT_VARIANCE_VS_DEPTH = "gradient_variance_vs_depth"


def filter_finished_pipeline_equals(pipeline_value: str) -> str:
    """Return MLflow ``filter_string`` for FINISHED runs with a given ``pipeline`` param."""
    return (
        f"attributes.status = 'FINISHED' and params.{ParamKey.PIPELINE} = '{pipeline_value}'"
    )


# Pre-built filters for common sweep resumes
FILTER_FINISHED_QSVM = filter_finished_pipeline_equals(PipelineValue.QSVM)


__all__ = [
    "FILTER_FINISHED_QSVM",
    "MetricKey",
    "ParamKey",
    "PIPELINE_ARCHITECTURE_DIAGNOSTICS",
    "PipelineValue",
    "RegimeValue",
    "TASK_EXPRESSIBILITY_BATCH",
    "TASK_GRADIENT_VARIANCE_VS_DEPTH",
    "filter_finished_pipeline_equals",
]
