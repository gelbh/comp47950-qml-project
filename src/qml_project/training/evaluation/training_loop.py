"""COBYLA training loop and held-out classifier evaluation."""

from __future__ import annotations

import time
import warnings
from contextlib import nullcontext
from typing import Any

import numpy as np
from qiskit.primitives import StatevectorSampler
from scipy.optimize import minimize

from qml_project.circuit import VariationalClassifier
from qml_project.training.mlflow_helpers import set_mlflow_tracking_uri
from qml_project.training.types import (
    DecisionRule,
    LossName,
    MeasurementObservable,
    TrainingHistory,
)

from .circuits import (
    _loss_from_outputs,
    _predict_from_outputs,
    evaluate_circuit_outputs,
    shots_for_eval,
)


def train_classifier(
    vc: VariationalClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    *,
    max_iter: int = 200,
    shot_schedule: dict[int, int] | None = None,
    seed: int = 42,
    test_shots: int = 300,
    sampler: Any | None = None,
    observable: MeasurementObservable = "bitstring_probs",
    decision_rule: DecisionRule = "argmax",
    loss_name: LossName = "softmax_nll",
    expectation_qubit: int = 0,
    verbose: bool = True,
    log_interval: int = 10,
    mlflow_experiment: str | None = None,
) -> tuple[np.ndarray, TrainingHistory]:
    r"""Train the variational classifier using COBYLA.

    Uses gradient-free COBYLA, the progressive shot schedule, and random
    initialisation in :math:`[-\pi, \pi]`.
    """
    rng = np.random.default_rng(seed)
    theta_init = rng.uniform(-np.pi, np.pi, vc.n_trainable)

    if sampler is None:
        sampler = StatevectorSampler(seed=seed)
    if loss_name != "softmax_nll" and vc.n_classes != 2:
        raise ValueError("Expectation-based losses require binary classes.")
    if decision_rule == "expectation_threshold" and vc.n_classes != 2:
        raise ValueError("Expectation-threshold decision rule requires binary classes.")
    if observable not in ("bitstring_probs", "z_expectation"):
        raise ValueError("observable must be 'bitstring_probs' or 'z_expectation'")

    history = TrainingHistory()
    eval_counter = [0]  # boxed for closure access

    mlflow_mod: Any = None
    mlflow_cm: Any = nullcontext()
    if mlflow_experiment:
        try:
            import mlflow as mlflow_mod
        except ImportError:
            if verbose:
                warnings.warn("MLflow not available, skipping logging", stacklevel=2)
        else:
            set_mlflow_tracking_uri()
            mlflow_mod.set_experiment(mlflow_experiment)
            mlflow_cm = mlflow_mod.start_run()

    with mlflow_cm:
        if mlflow_mod is not None:
            mlflow_mod.log_params({
                "seed": seed,
                "max_iter": max_iter,
                "n_qubits": vc.n_qubits,
                "n_features": vc.n_features,
                "n_classes": vc.n_classes,
                "n_trainable": vc.n_trainable,
                "ansatz": vc.ansatz,
                "test_shots": test_shots,
                "observable": observable,
                "decision_rule": decision_rule,
                "loss_name": loss_name,
                "expectation_qubit": expectation_qubit,
            })

        t0 = time.perf_counter()

        def objective(params: np.ndarray) -> float:
            eval_counter[0] += 1
            n_eval = eval_counter[0]
            shots = shots_for_eval(n_eval, shot_schedule)

            outputs = evaluate_circuit_outputs(
                vc,
                X_train,
                params,
                shots,
                sampler,
                expectation_qubit=expectation_qubit,
            )
            loss_val = _loss_from_outputs(outputs, y_train, loss_name=loss_name)

            if loss_val < history.best_loss:
                history.best_loss = loss_val
                history.best_weights = params.copy()

            if n_eval % log_interval == 0 or n_eval == 1:
                train_preds = _predict_from_outputs(outputs, decision_rule=decision_rule)
                train_acc = float(np.mean(train_preds == y_train))

                history.losses.append(loss_val)
                history.train_accuracies.append(train_acc)
                history.eval_numbers.append(n_eval)
                history.shot_counts.append(shots)

                if X_test is not None and y_test is not None:
                    test_outputs = evaluate_circuit_outputs(
                        vc,
                        X_test,
                        params,
                        test_shots,
                        sampler,
                        expectation_qubit=expectation_qubit,
                    )
                    test_preds = _predict_from_outputs(
                        test_outputs, decision_rule=decision_rule
                    )
                    test_acc = float(np.mean(test_preds == y_test))
                    history.test_accuracies.append(test_acc)

                if verbose:
                    msg = (
                        f"  Eval {n_eval:4d} | loss={loss_val:.4f}"
                        f" | train_acc={train_acc:.3f}"
                    )
                    if history.test_accuracies:
                        msg += f" | test_acc={history.test_accuracies[-1]:.3f}"
                    msg += f" | shots={shots}"
                    print(msg)

            return loss_val

        opt_result = minimize(
            objective,
            theta_init,
            method="COBYLA",
            options={"maxiter": max_iter, "rhobeg": 0.5},
        )

        history.total_training_time = time.perf_counter() - t0
        history.total_evals = eval_counter[0]

        best_weights: np.ndarray = (
            history.best_weights
            if history.best_weights is not None
            else opt_result.x
        )
        history.best_weights = best_weights

        if mlflow_mod is not None:
            try:
                mlflow_mod.log_metrics({
                    "train_accuracy": history.train_accuracies[-1] if history.train_accuracies else 0.0,
                    "test_accuracy": history.test_accuracies[-1] if history.test_accuracies else 0.0,
                    "final_loss": history.best_loss,
                    "training_time": history.total_training_time,
                    "total_evals": history.total_evals,
                })
            except Exception as exc:
                warnings.warn(f"MLflow logging failed: {exc}", stacklevel=2)

    if verbose:
        print(
            f"\nTraining complete in {history.total_training_time:.1f}s "
            f"({history.total_evals} evaluations)"
        )
        print(f"Best loss: {history.best_loss:.4f}")
        print(f"COBYLA status: {opt_result.message}")

    return best_weights, history


def evaluate_classifier(
    vc: VariationalClassifier,
    X: np.ndarray,
    y: np.ndarray,
    theta: np.ndarray,
    *,
    shots: int = 300,
    sampler: Any | None = None,
    seed: int = 42,
    decision_rule: DecisionRule = "argmax",
    expectation_qubit: int = 0,
) -> dict:
    """Evaluate a trained classifier on held-out data.

    Returns
    -------
    dict
        Keys: ``accuracy``, ``predictions``, ``class_probs``,
        ``z_expectations``, ``inference_time``.
    """
    if sampler is None:
        sampler = StatevectorSampler(seed=seed)

    t0 = time.perf_counter()
    outputs = evaluate_circuit_outputs(
        vc,
        X,
        theta,
        shots,
        sampler,
        expectation_qubit=expectation_qubit,
    )
    inference_time = time.perf_counter() - t0

    preds = _predict_from_outputs(outputs, decision_rule=decision_rule)
    accuracy = float(np.mean(preds == y))

    return {
        "accuracy": accuracy,
        "predictions": preds,
        "class_probs": outputs["class_probs"],
        "z_expectations": outputs["z_expectations"],
        "inference_time": inference_time,
    }
