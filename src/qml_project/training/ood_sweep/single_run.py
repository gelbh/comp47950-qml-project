"""Single VQC OOD train/eval point used by the sweep workers."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from qml_project.circuit import VariationalClassifier
from qml_project.training.evaluation import (
    evaluate_classifier,
    evaluate_vqc_win_rate,
    train_classifier,
)
from qml_project.training.metrics import _metrics_from_preds
from qml_project.training.types import (
    DecisionRule,
    LossName,
    MeasurementObservable,
    SimulatedVQCRunResult,
)


def simulated_vqc_ood_single_run(
    vc: VariationalClassifier,
    subset_X: np.ndarray,
    subset_y: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    train_size: int,
    seed: int,
    max_iter: int,
    shot_schedule: dict[int, int] | None,
    test_shots: int,
    sampler: Any | None,
    decision_rule: DecisionRule,
    observable: MeasurementObservable,
    loss_name: LossName,
    expectation_qubit: int,
    feature_fn_for_policy: Callable[[np.ndarray], np.ndarray] | None,
    compute_win_rate: bool,
    n_games_win_rate: int,
    game_k: int,
    game_M: int,
    train_verbose: bool,
    log_interval: int,
) -> SimulatedVQCRunResult:
    """Train and evaluate one VQC on a train subset (OOD protocol)."""
    best_weights, history = train_classifier(
        vc,
        subset_X,
        subset_y,
        X_test,
        y_test,
        max_iter=max_iter,
        shot_schedule=shot_schedule,
        seed=int(seed),
        test_shots=test_shots,
        sampler=sampler,
        observable=observable,
        decision_rule=decision_rule,
        loss_name=loss_name,
        expectation_qubit=expectation_qubit,
        verbose=train_verbose,
        log_interval=log_interval,
        mlflow_experiment=None,
    )
    eval_result = evaluate_classifier(
        vc,
        X_test,
        y_test,
        best_weights,
        shots=test_shots,
        sampler=sampler,
        seed=int(seed),
        decision_rule=decision_rule,
        expectation_qubit=expectation_qubit,
    )
    test_acc, bal_acc, mcc_val = _metrics_from_preds(
        y_test, eval_result["predictions"]
    )
    win_rate_val: float | None = None
    if compute_win_rate and feature_fn_for_policy is not None:
        win_rate_val = evaluate_vqc_win_rate(
            vc,
            best_weights,
            feature_fn_for_policy,
            n_games=n_games_win_rate,
            k=game_k,
            M=game_M,
            seed=int(seed),
            shots=test_shots,
            sampler=sampler,
            decision_rule=decision_rule,
            expectation_qubit=expectation_qubit,
        )
    return SimulatedVQCRunResult(
        train_size=int(train_size),
        seed=int(seed),
        test_accuracy=test_acc,
        balanced_accuracy=bal_acc,
        mcc=mcc_val,
        win_rate=win_rate_val,
        training_time=float(history.total_training_time),
        inference_time=float(eval_result["inference_time"]),
        final_loss=float(history.best_loss),
        ansatz=str(vc.ansatz),
        observable=observable,
        decision_rule=decision_rule,
        loss_name=loss_name,
    )


__all__ = ["simulated_vqc_ood_single_run"]
