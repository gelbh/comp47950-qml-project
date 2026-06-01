"""MLflow cache loader and per-run logger for the VQC noise sweep."""

from __future__ import annotations

import contextlib

from qml_project.circuit import VariationalClassifier
from qml_project.training.mlflow_schema import MetricKey, ParamKey, PipelineValue
from qml_project.training.types import (
    DecisionRule,
    LossName,
    MeasurementObservable,
    VqcNoiseSweepRunResult,
)


def _log_vqc_noise_result_to_mlflow(
    r: VqcNoiseSweepRunResult,
    *,
    mlflow_run_prefix: str,
    probe_vc: VariationalClassifier,
    max_iter: int,
    decision_rule: DecisionRule,
    observable: MeasurementObservable,
    loss_name: LossName,
    expectation_qubit: int,
) -> None:
    """Log one noise-sweep grid point as a flat MLflow run."""
    import mlflow

    run_name = (
        f"{mlflow_run_prefix}|{r.noise_profile}|"
        f"lvl={r.noise_level if r.noise_level is not None else 'none'}|"
        f"shots={r.shots}|seed={r.seed}"
    )
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                ParamKey.PIPELINE: PipelineValue.SIMULATED_VQC_NOISE,
                ParamKey.RUN_PREFIX: mlflow_run_prefix,
                ParamKey.NOISE_PROFILE: r.noise_profile,
                ParamKey.NOISE_LEVEL: "none" if r.noise_level is None else float(r.noise_level),
                ParamKey.SHOTS: int(r.shots),
                ParamKey.SEED: int(r.seed),
                ParamKey.ANSATZ: r.ansatz,
                ParamKey.N_QUBITS: probe_vc.n_qubits,
                ParamKey.N_FEATURES: probe_vc.n_features,
                ParamKey.N_TRAINABLE: probe_vc.n_trainable,
                ParamKey.MAX_ITER: int(max_iter),
                ParamKey.DECISION_RULE: decision_rule,
                ParamKey.OBSERVABLE: observable,
                ParamKey.LOSS_NAME: loss_name,
                ParamKey.EXPECTATION_QUBIT: int(expectation_qubit),
            }
        )
        metrics: dict[str, float] = {
            MetricKey.TRAINING_TIME: float(r.training_time),
            MetricKey.INFERENCE_TIME: float(r.inference_time),
            MetricKey.FINAL_LOSS: float(r.final_loss),
            "test_accuracy_raw": float(r.test_accuracy_raw),
            "balanced_accuracy_raw": float(r.balanced_accuracy_raw),
            "mcc_raw": float(r.mcc_raw),
        }
        if r.test_accuracy_readout is not None:
            metrics["test_accuracy_readout"] = float(r.test_accuracy_readout)
        if r.balanced_accuracy_readout is not None:
            metrics["balanced_accuracy_readout"] = float(r.balanced_accuracy_readout)
        if r.mcc_readout is not None:
            metrics["mcc_readout"] = float(r.mcc_readout)
        if r.test_accuracy_zne is not None:
            metrics["test_accuracy_zne"] = float(r.test_accuracy_zne)
        if r.balanced_accuracy_zne is not None:
            metrics["balanced_accuracy_zne"] = float(r.balanced_accuracy_zne)
        if r.mcc_zne is not None:
            metrics["mcc_zne"] = float(r.mcc_zne)
        if r.test_accuracy_readout_zne is not None:
            metrics["test_accuracy_readout_zne"] = float(r.test_accuracy_readout_zne)
        if r.balanced_accuracy_readout_zne is not None:
            metrics["balanced_accuracy_readout_zne"] = float(
                r.balanced_accuracy_readout_zne
            )
        if r.mcc_readout_zne is not None:
            metrics["mcc_readout_zne"] = float(r.mcc_readout_zne)
        mlflow.log_metrics(metrics)


def _load_vqc_noise_sweep_from_mlflow(
    experiment_name: str,
    mlflow_run_prefix: str,
    *,
    probe_ansatz: str,
) -> dict[tuple[str, float | None, int, int], VqcNoiseSweepRunResult]:
    """Load finished noise-sweep runs keyed by ``(profile, level, shots, seed)``."""
    cache: dict[tuple[str, float | None, int, int], VqcNoiseSweepRunResult] = {}
    try:
        from mlflow.tracking import MlflowClient
    except ImportError:
        return cache

    with contextlib.suppress(Exception):
        client = MlflowClient()
        exp = client.get_experiment_by_name(experiment_name)
        if exp is None:
            return cache
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["end_time DESC"],
            max_results=20_000,
        )
        for run in runs:
            if run.info.status != "FINISHED":
                continue
            p = run.data.params
            m = run.data.metrics
            if p.get(ParamKey.PIPELINE) != PipelineValue.SIMULATED_VQC_NOISE:
                continue
            if p.get(ParamKey.RUN_PREFIX) != mlflow_run_prefix:
                continue
            if p.get(ParamKey.ANSATZ) != probe_ansatz:
                continue
            try:
                prof = p[ParamKey.NOISE_PROFILE]
                level = (
                    None
                    if p.get(ParamKey.NOISE_LEVEL) in (None, "none")
                    else float(p[ParamKey.NOISE_LEVEL])
                )
                shots = int(p[ParamKey.SHOTS])
                seed = int(p[ParamKey.SEED])
            except (KeyError, TypeError, ValueError):
                continue
            key = (prof, level, shots, seed)
            if key in cache:
                continue
            cache[key] = VqcNoiseSweepRunResult(
                noise_profile=prof,
                noise_level=level,
                shots=shots,
                seed=seed,
                ansatz=probe_ansatz,
                training_time=float(m.get(MetricKey.TRAINING_TIME, 0.0)),
                inference_time=float(m.get(MetricKey.INFERENCE_TIME, 0.0)),
                final_loss=float(m.get(MetricKey.FINAL_LOSS, 0.0)),
                test_accuracy_raw=float(m.get("test_accuracy_raw", 0.0)),
                balanced_accuracy_raw=float(m.get("balanced_accuracy_raw", 0.0)),
                mcc_raw=float(m.get("mcc_raw", 0.0)),
                test_accuracy_readout=(
                    float(m["test_accuracy_readout"])
                    if "test_accuracy_readout" in m
                    else None
                ),
                balanced_accuracy_readout=(
                    float(m["balanced_accuracy_readout"])
                    if "balanced_accuracy_readout" in m
                    else None
                ),
                mcc_readout=float(m["mcc_readout"]) if "mcc_readout" in m else None,
                test_accuracy_zne=(
                    float(m["test_accuracy_zne"]) if "test_accuracy_zne" in m else None
                ),
                balanced_accuracy_zne=(
                    float(m["balanced_accuracy_zne"])
                    if "balanced_accuracy_zne" in m
                    else None
                ),
                mcc_zne=float(m["mcc_zne"]) if "mcc_zne" in m else None,
                test_accuracy_readout_zne=(
                    float(m["test_accuracy_readout_zne"])
                    if "test_accuracy_readout_zne" in m
                    else None
                ),
                balanced_accuracy_readout_zne=(
                    float(m["balanced_accuracy_readout_zne"])
                    if "balanced_accuracy_readout_zne" in m
                    else None
                ),
                mcc_readout_zne=(
                    float(m["mcc_readout_zne"]) if "mcc_readout_zne" in m else None
                ),
            )
    return cache


__all__ = [
    "_load_vqc_noise_sweep_from_mlflow",
    "_log_vqc_noise_result_to_mlflow",
]
