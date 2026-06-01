"""Param and metric key conventions for the architecture-diagnostics cache.

Formatting helpers centralise string shapes logged to MLflow. Scalar constants
(:data:`PIPELINE`, task names) live in :mod:`qml_project.training.mlflow_schema`.
"""

from __future__ import annotations

from typing import Sequence

from qml_project.training.mlflow_schema import (
    PIPELINE_ARCHITECTURE_DIAGNOSTICS,
    TASK_EXPRESSIBILITY_BATCH,
    TASK_GRADIENT_VARIANCE_VS_DEPTH,
)

PIPELINE = PIPELINE_ARCHITECTURE_DIAGNOSTICS
TASK_EXPRESS = TASK_EXPRESSIBILITY_BATCH
TASK_GRAD = TASK_GRADIENT_VARIANCE_VS_DEPTH


def depth_ladder_param(depths: Sequence[int]) -> str:
    return ",".join(str(int(d)) for d in depths)


def encoding_specs_param(specs: Sequence[tuple[str, int, int]]) -> str:
    return "|".join(
        f"{enc}:{nq}:{nf}" for enc, nq, nf in sorted(specs, key=lambda t: t[0])
    )


def ansatze_param(ansatze: Sequence[str]) -> str:
    return ",".join(str(a) for a in ansatze)


def metric_key_grad_mean(depth: int) -> str:
    return f"gradient_variance_mean_d{int(depth)}"


def metric_key_grad_std(depth: int) -> str:
    return f"gradient_variance_std_d{int(depth)}"


def metric_key_kl(ansatz: str) -> str:
    return f"kl_{ansatz}"


def metric_key_mw_mean(ansatz: str) -> str:
    return f"mw_mean_{ansatz}"


def metric_key_mw_std(ansatz: str) -> str:
    return f"mw_std_{ansatz}"
