"""Per-run MLflow logger for the simulated-VQC OOD sweep.

The OOD cache loader lives in :mod:`qml_project.training.mlflow_helpers`
(``_load_simulated_vqc_ood_from_mlflow``) since it shares schema knowledge
with other VQC pipelines; we just import the logger from here.
"""

from __future__ import annotations

from typing import Any

from qml_project.circuit import VariationalClassifier
from qml_project.training.experiment_namespace import standard_tags
from qml_project.training.mlflow_helpers import log_training_run
from qml_project.training.types import (
    DecisionRule,
    LossName,
    MeasurementObservable,
    SimulatedVQCRunResult,
)


def _log_simulated_vqc_ood_result_to_mlflow(
    res: SimulatedVQCRunResult,
    *,
    mlflow_run_prefix: str,
    vc_meta: VariationalClassifier,
    max_iter: int,
    test_shots: int,
    observable: MeasurementObservable,
    decision_rule: DecisionRule,
    loss_name: LossName,
    expectation_qubit: int,
    n_games_win_rate: int,
    extra_params: dict[str, Any] | None = None,
) -> None:
    import mlflow

    run_name = f"{mlflow_run_prefix}|n={res.train_size}|s={res.seed}"
    xp = dict(extra_params or {})
    enc = xp.get("encoding")
    cfg = xp.get("config_id")
    inc_tag: bool | None
    if "include_nim_sum" in xp:
        inc_tag = bool(xp["include_nim_sum"])
    else:
        inc_tag = None
    tag_extra = {
        k: xp[k]
        for k in ("n_layers", "cz_strategy")
        if k in xp and xp[k] is not None
    }
    tags = standard_tags(
        pipeline="vqc",
        stage="ood",
        encoding=str(enc) if enc is not None else None,
        train_size=res.train_size,
        seed=int(res.seed),
        ansatz=str(res.ansatz),
        loss_name=str(loss_name),
        include_nim_sum=inc_tag,
        config_id=str(cfg) if cfg not in (None, "") else None,
        extra=tag_extra if tag_extra else None,
    )
    params: dict[str, Any] = {
        "pipeline": "simulated_vqc",
        "regime": "ood",
        "train_size": res.train_size,
        "seed": int(res.seed),
        "max_iter": max_iter,
        "test_shots": test_shots,
        "ansatz": res.ansatz,
        "n_qubits": vc_meta.n_qubits,
        "n_features": vc_meta.n_features,
        "n_trainable": vc_meta.n_trainable,
        "observable": observable,
        "decision_rule": decision_rule,
        "loss_name": loss_name,
        "expectation_qubit": expectation_qubit,
        "n_games_win_rate": n_games_win_rate,
    }
    if extra_params:
        params.update(extra_params)
    metrics: dict[str, float] = {
        "test_accuracy": float(res.test_accuracy),
        "balanced_accuracy": float(res.balanced_accuracy),
        "mcc": float(res.mcc),
        "training_time": float(res.training_time),
        "inference_time": float(res.inference_time),
        "final_loss": float(res.final_loss),
    }
    if res.win_rate is not None:
        metrics["win_rate"] = float(res.win_rate)
    log_training_run(
        mlflow,
        run_name=run_name,
        params=params,
        metrics=metrics,
        tags=tags,
    )


__all__ = ["_log_simulated_vqc_ood_result_to_mlflow"]
