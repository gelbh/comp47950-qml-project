"""Multi-seed training experiments with optional MLflow logging."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from qml_project.circuit import VariationalClassifier
from qml_project.training.evaluation import evaluate_classifier, train_classifier
from qml_project.training.metrics import _metrics_from_preds
from qml_project.training.mlflow_helpers import (
    _load_multi_seed_summary_from_mlflow,
    _parent_run_param_signature,
    _set_mlflow_tracking_uri,
)
from qml_project.training.types import (
    DecisionRule,
    ExperimentResult,
    LossName,
    MeasurementObservable,
    MultiSeedSummary,
)


def run_multi_seed_experiment(
    vc_builder: Callable[[], VariationalClassifier],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    seeds: list[int] | None = None,
    n_seeds: int = 5,
    max_iter: int = 200,
    shot_schedule: dict[int, int] | None = None,
    test_shots: int = 300,
    sampler_factory: Callable[[int], Any] | None = None,
    decision_rule: DecisionRule = "argmax",
    observable: MeasurementObservable = "bitstring_probs",
    loss_name: LossName = "softmax_nll",
    expectation_qubit: int = 0,
    verbose: bool = True,
    log_interval: int = 20,
    mlflow_experiment: str | None = None,
    mlflow_run_name: str | None = None,
    use_cache: bool = True,
    force_rerun: bool = False,
) -> MultiSeedSummary:
    """
    Train with multiple random seeds and aggregate results.

    This directly addresses QML model volatility: models can be quite
    volatile and sensitive to starting conditions, so multiple seeds
    are recommended.

    Parameters
    ----------
    vc_builder : callable
        Zero-argument callable returning a fresh ``VariationalClassifier``.
    seeds : list[int] or None
        Explicit seeds.  If *None*, uses ``list(range(n_seeds))``.
    sampler_factory : callable or None
        ``seed -> sampler``.  If *None*, uses ``StatevectorSampler``.
    mlflow_run_name : str or None
        Parent run name; used with ``mlflow_experiment`` for MLflow cache
        lookup when ``use_cache=True``.
    use_cache : bool
        If True and ``mlflow_experiment`` and ``mlflow_run_name`` are set,
        load matching finished runs from MLflow instead of training.
    force_rerun : bool
        If True, always train and log; ignore MLflow cache.
    """
    if seeds is None:
        seeds = list(range(n_seeds))

    temp_vc = vc_builder()

    if (
        use_cache
        and not force_rerun
        and mlflow_experiment
        and mlflow_run_name
    ):
        cached = _load_multi_seed_summary_from_mlflow(
            mlflow_experiment,
            mlflow_run_name,
            seeds=seeds,
            max_iter=max_iter,
            test_shots=test_shots,
            n_qubits=temp_vc.n_qubits,
            n_features=temp_vc.n_features,
            n_classes=temp_vc.n_classes,
            n_trainable=temp_vc.n_trainable,
            ansatz=str(temp_vc.ansatz),
            observable=observable,
            decision_rule=decision_rule,
            loss_name=loss_name,
            expectation_qubit=expectation_qubit,
            verbose=verbose,
        )
        if cached is not None:
            return cached

    mlflow_parent_run = None
    if mlflow_experiment:
        try:
            import mlflow

            _set_mlflow_tracking_uri()
            mlflow.set_experiment(mlflow_experiment)
            mlflow_parent_run = mlflow.start_run(run_name=mlflow_run_name)
            mlflow.log_params(
                _parent_run_param_signature(
                    seeds=seeds,
                    max_iter=max_iter,
                    test_shots=test_shots,
                    n_qubits=temp_vc.n_qubits,
                    n_features=temp_vc.n_features,
                    n_classes=temp_vc.n_classes,
                    n_trainable=temp_vc.n_trainable,
                    ansatz=str(temp_vc.ansatz),
                    observable=observable,
                    decision_rule=decision_rule,
                    loss_name=loss_name,
                    expectation_qubit=expectation_qubit,
                )
            )
        except ImportError:
            if verbose:
                print("Warning: MLflow not available, skipping logging")
            mlflow_parent_run = None

    all_results: list[ExperimentResult] = []

    for i, seed in enumerate(seeds):
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Seed {seed} ({i + 1}/{len(seeds)})")
            print(f"{'=' * 60}")

        if mlflow_parent_run:
            try:
                import mlflow

                mlflow.start_run(run_name=f"seed_{seed}", nested=True)
            except Exception:
                pass

        vc = vc_builder()
        sampler = (
            sampler_factory(seed) if sampler_factory is not None else None
        )

        best_weights, history = train_classifier(
            vc,
            X_train,
            y_train,
            X_test,
            y_test,
            max_iter=max_iter,
            shot_schedule=shot_schedule,
            seed=seed,
            test_shots=test_shots,
            sampler=sampler,
            observable=observable,
            decision_rule=decision_rule,
            loss_name=loss_name,
            expectation_qubit=expectation_qubit,
            verbose=verbose,
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
            seed=seed,
            decision_rule=decision_rule,
            expectation_qubit=expectation_qubit,
        )

        if mlflow_parent_run:
            try:
                import mlflow

                _, bal_acc, mcc_val = _metrics_from_preds(
                    y_test, eval_result["predictions"]
                )
                mlflow.log_params({"seed": seed})
                mlflow.log_metrics({
                    "train_accuracy": history.train_accuracies[-1] if history.train_accuracies else 0.0,
                    "test_accuracy": eval_result["accuracy"],
                    "training_time": history.total_training_time,
                    "final_loss": history.best_loss,
                    "balanced_accuracy": bal_acc,
                    "mcc": mcc_val,
                    "inference_time": eval_result["inference_time"],
                })
                mlflow.end_run()
            except Exception:
                pass

        all_results.append(
            ExperimentResult(
                seed=seed,
                best_weights=best_weights,
                history=history,
                test_accuracy=eval_result["accuracy"],
                test_predictions=eval_result["predictions"],
                test_class_probs=eval_result["class_probs"],
                training_time=history.total_training_time,
                inference_time=eval_result["inference_time"],
            )
        )

    test_accs = [r.test_accuracy for r in all_results]
    train_times = [r.training_time for r in all_results]
    inference_times = [r.inference_time for r in all_results]

    summary = MultiSeedSummary(
        per_seed=all_results,
        test_accuracy_mean=float(np.mean(test_accs)),
        test_accuracy_std=float(np.std(test_accs)),
        test_accuracy_min=float(np.min(test_accs)),
        test_accuracy_max=float(np.max(test_accs)),
        training_time_mean=float(np.mean(train_times)),
        inference_time_mean=float(np.mean(inference_times)),
        n_seeds=len(seeds),
    )

    if mlflow_parent_run:
        try:
            import mlflow

            mlflow.log_metrics({
                "mean_test_accuracy": summary.test_accuracy_mean,
                "std_test_accuracy": summary.test_accuracy_std,
                "min_test_accuracy": summary.test_accuracy_min,
                "max_test_accuracy": summary.test_accuracy_max,
                "mean_training_time": summary.training_time_mean,
                "mean_inference_time": summary.inference_time_mean,
            })
            mlflow.end_run()
        except Exception as e:
            if verbose:
                print(f"Warning: MLflow logging failed: {e}")

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"SUMMARY ({len(seeds)} seeds)")
        print(f"{'=' * 60}")
        print(
            f"Test accuracy: {summary.test_accuracy_mean:.4f}"
            f" ± {summary.test_accuracy_std:.4f}"
        )
        print(
            f"  Range: [{summary.test_accuracy_min:.4f},"
            f" {summary.test_accuracy_max:.4f}]"
        )
        print(f"Mean training time: {summary.training_time_mean:.1f}s")
        print(f"Mean inference time: {summary.inference_time_mean:.3f}s")

    return summary
