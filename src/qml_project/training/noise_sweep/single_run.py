"""Single VQC noise-sweep training+evaluation point (pure helper)."""

from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np

from qml_project.circuit import VariationalClassifier
from qml_project.training.evaluation import evaluate_circuit_outputs, train_classifier
from qml_project.training.evaluation.circuits import _predict_from_outputs
from qml_project.training.metrics import _metrics_from_preds
from qml_project.training.noise_aer import (
    build_assignment_matrix_from_symmetric_readout_error,
    create_depolarizing_noise_model,
    create_noisy_sampler,
    _zne_extrapolate_outputs,
)
from qml_project.training.types import (
    DecisionRule,
    LossName,
    MeasurementObservable,
    VqcNoiseSweepRunResult,
    VqcNoiseSweepTask,
)


def _single_noise_run(
    *,
    vc: VariationalClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task: VqcNoiseSweepTask,
    max_iter: int,
    shot_schedule: dict[int, int] | None,
    decision_rule: DecisionRule,
    observable: MeasurementObservable,
    loss_name: LossName,
    expectation_qubit: int,
    zne_scales: Sequence[float],
    zne_degree: int,
    single_gate_error_ratio: float,
    readout_error_rate: float,
    backend_noise_model: Any | None,
    apply_readout_correction: bool,
    apply_zne: bool,
    log_interval: int,
) -> VqcNoiseSweepRunResult:
    """Train/evaluate one point of the noise sweep."""
    seed = int(task.seed)
    if task.noise_profile == "depolarizing":
        base_rate = float(task.noise_level or 0.0)
        noise_model = create_depolarizing_noise_model(
            cz_error_rate=base_rate,
            single_gate_error_rate=single_gate_error_ratio * base_rate,
            readout_error_rate=readout_error_rate,
        )
    elif task.noise_profile == "backend":
        if backend_noise_model is None:
            raise ValueError("backend_noise_model is required for backend profile.")
        noise_model = backend_noise_model
    else:
        raise ValueError(f"Unsupported noise profile: {task.noise_profile}")

    sampler = create_noisy_sampler(noise_model, seed=seed)
    best_weights, history = train_classifier(
        vc,
        X_train,
        y_train,
        X_test,
        y_test,
        max_iter=max_iter,
        shot_schedule=shot_schedule,
        seed=seed,
        test_shots=int(task.shots),
        sampler=sampler,
        observable=observable,
        decision_rule=decision_rule,
        loss_name=loss_name,
        expectation_qubit=expectation_qubit,
        verbose=False,
        log_interval=log_interval,
        mlflow_experiment=None,
    )

    t0 = time.perf_counter()
    base_outputs = evaluate_circuit_outputs(
        vc,
        X_test,
        best_weights,
        int(task.shots),
        sampler,
        expectation_qubit=expectation_qubit,
    )
    inference_time = time.perf_counter() - t0
    raw_preds = _predict_from_outputs(base_outputs, decision_rule=decision_rule)
    acc_raw, bal_raw, mcc_raw = _metrics_from_preds(y_test, raw_preds)

    readout_matrix: np.ndarray | None = None
    if apply_readout_correction:
        if task.noise_profile == "depolarizing" and readout_error_rate > 0:
            readout_matrix = build_assignment_matrix_from_symmetric_readout_error(
                n_qubits=vc.n_qubits,
                readout_error_rate=readout_error_rate,
            )

    readout_metrics: tuple[float, float, float] | None = None
    if readout_matrix is not None:
        readout_outputs = evaluate_circuit_outputs(
            vc,
            X_test,
            best_weights,
            int(task.shots),
            sampler,
            expectation_qubit=expectation_qubit,
            readout_assignment_matrix=readout_matrix,
        )
        readout_preds = _predict_from_outputs(readout_outputs, decision_rule=decision_rule)
        readout_metrics = _metrics_from_preds(y_test, readout_preds)

    zne_metrics: tuple[float, float, float] | None = None
    readout_zne_metrics: tuple[float, float, float] | None = None
    if apply_zne and task.noise_profile == "depolarizing":
        outputs_by_scale: list[dict[str, np.ndarray]] = []
        outputs_by_scale_readout: list[dict[str, np.ndarray]] = []
        for scale in zne_scales:
            scaled_rate = min(float(task.noise_level or 0.0) * float(scale), 0.49)
            scaled_noise = create_depolarizing_noise_model(
                cz_error_rate=scaled_rate,
                single_gate_error_rate=single_gate_error_ratio * scaled_rate,
                readout_error_rate=readout_error_rate,
            )
            scaled_sampler = create_noisy_sampler(scaled_noise, seed=seed)
            outputs_by_scale.append(
                evaluate_circuit_outputs(
                    vc,
                    X_test,
                    best_weights,
                    int(task.shots),
                    scaled_sampler,
                    expectation_qubit=expectation_qubit,
                )
            )
            if readout_matrix is not None:
                outputs_by_scale_readout.append(
                    evaluate_circuit_outputs(
                        vc,
                        X_test,
                        best_weights,
                        int(task.shots),
                        scaled_sampler,
                        expectation_qubit=expectation_qubit,
                        readout_assignment_matrix=readout_matrix,
                    )
                )

        zne_outputs = _zne_extrapolate_outputs(
            outputs_by_scale,
            scales=zne_scales,
            degree=zne_degree,
        )
        zne_preds = _predict_from_outputs(zne_outputs, decision_rule=decision_rule)
        zne_metrics = _metrics_from_preds(y_test, zne_preds)

        if outputs_by_scale_readout:
            zne_outputs_readout = _zne_extrapolate_outputs(
                outputs_by_scale_readout,
                scales=zne_scales,
                degree=zne_degree,
            )
            zne_preds_readout = _predict_from_outputs(
                zne_outputs_readout, decision_rule=decision_rule
            )
            readout_zne_metrics = _metrics_from_preds(y_test, zne_preds_readout)

    return VqcNoiseSweepRunResult(
        noise_profile=task.noise_profile,
        noise_level=task.noise_level,
        shots=int(task.shots),
        seed=seed,
        ansatz=str(vc.ansatz),
        training_time=float(history.total_training_time),
        inference_time=float(inference_time),
        final_loss=float(history.best_loss),
        test_accuracy_raw=acc_raw,
        balanced_accuracy_raw=bal_raw,
        mcc_raw=mcc_raw,
        test_accuracy_readout=None if readout_metrics is None else readout_metrics[0],
        balanced_accuracy_readout=None if readout_metrics is None else readout_metrics[1],
        mcc_readout=None if readout_metrics is None else readout_metrics[2],
        test_accuracy_zne=None if zne_metrics is None else zne_metrics[0],
        balanced_accuracy_zne=None if zne_metrics is None else zne_metrics[1],
        mcc_zne=None if zne_metrics is None else zne_metrics[2],
        test_accuracy_readout_zne=None
        if readout_zne_metrics is None
        else readout_zne_metrics[0],
        balanced_accuracy_readout_zne=None
        if readout_zne_metrics is None
        else readout_zne_metrics[1],
        mcc_readout_zne=None if readout_zne_metrics is None else readout_zne_metrics[2],
    )


__all__ = ["_single_noise_run"]
