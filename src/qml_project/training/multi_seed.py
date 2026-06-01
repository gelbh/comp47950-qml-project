"""Multi-seed training experiments with optional MLflow logging."""

from __future__ import annotations

import contextlib
import warnings
from typing import Any, Callable

import numpy as np

from qml_project.circuit import VariationalClassifier
from qml_project.training.evaluation import evaluate_classifier, train_classifier
from qml_project.training.metrics import _metrics_from_preds
from qml_project.training.mlflow_helpers import (
    _load_multi_seed_summary_from_mlflow,
    _partial_multi_seed_cached_experiment_results,
    _parent_run_param_signature,
    set_mlflow_tracking_uri,
    end_mlflow_run_if_active_id,
    end_mlflow_run_if_nested_under,
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
) -> MultiSeedSummary:
    """
    Train one VQC with multiple random seeds and aggregate the per-seed metrics.

    VQC training is sensitive to the starting parameters, so means and spreads
    over several seeds give a more honest picture than a single fitted model.

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
        load finished nested seed runs from MLflow instead of re-training those
        seeds. Children from **any** prior parent with the same run name and
        params are merged (newest parent wins per seed), so you can interrupt
        mid-run and resume: completed seeds stay cached; only missing seeds
        train and log under a new parent run. Set False to always train fresh.
    """
    if seeds is None:
        seeds = list(range(n_seeds))

    temp_vc = vc_builder()

    cached_by_seed: dict[int, ExperimentResult] = {}
    if use_cache and mlflow_experiment and mlflow_run_name:
        full_summary = _load_multi_seed_summary_from_mlflow(
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
        if full_summary is not None:
            return full_summary
        cached_by_seed = _partial_multi_seed_cached_experiment_results(
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
        )
        if verbose and cached_by_seed:
            print(
                f"  Resuming {len(cached_by_seed)}/{len(seeds)} seed(s) from MLflow "
                f"(run_name={mlflow_run_name!r}); training the rest."
            )

    missing_seeds = [s for s in seeds if s not in cached_by_seed]
    if not missing_seeds:
        per_seed = [cached_by_seed[s] for s in seeds]
        test_accs = [r.test_accuracy for r in per_seed]
        train_times = [r.training_time for r in per_seed]
        inference_times = [r.inference_time for r in per_seed]
        summary = MultiSeedSummary(
            per_seed=per_seed,
            test_accuracy_mean=float(np.mean(test_accs)),
            test_accuracy_std=float(np.std(test_accs)),
            test_accuracy_min=float(np.min(test_accs)),
            test_accuracy_max=float(np.max(test_accs)),
            training_time_mean=float(np.mean(train_times)),
            inference_time_mean=float(np.mean(inference_times)),
            n_seeds=len(seeds),
        )
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"SUMMARY ({len(seeds)} seeds, all from MLflow cache)")
            print(f"{'=' * 60}")
            print(
                f"Test accuracy: {summary.test_accuracy_mean:.4f}"
                f" ± {summary.test_accuracy_std:.4f}"
            )
        return summary

    mlflow_parent_run = None
    parent_run_id: str | None = None
    if mlflow_experiment and missing_seeds:
        try:
            import mlflow

            set_mlflow_tracking_uri()
            mlflow.set_experiment(mlflow_experiment)
            mlflow_parent_run = mlflow.start_run(run_name=mlflow_run_name)
            parent_run_id = mlflow.active_run().info.run_id
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
            parent_run_id = None

    all_results: list[ExperimentResult] = []

    try:
        for i, seed in enumerate(seeds):
            if seed in cached_by_seed:
                all_results.append(cached_by_seed[seed])
                if verbose:
                    print(f"\n{'=' * 60}")
                    print(
                        f"Seed {seed} ({i + 1}/{len(seeds)}) — loaded from MLflow cache"
                    )
                    print(f"{'=' * 60}")
                continue

            if verbose:
                print(f"\n{'=' * 60}")
                print(f"Seed {seed} ({i + 1}/{len(seeds)})")
                print(f"{'=' * 60}")

            if mlflow_parent_run:
                with contextlib.suppress(Exception):
                    import mlflow

                    mlflow.start_run(run_name=f"seed_{seed}", nested=True)

            try:
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
                    with contextlib.suppress(Exception):
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
            finally:
                if parent_run_id is not None:
                    end_mlflow_run_if_nested_under(parent_run_id)

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
            except Exception as exc:
                if verbose:
                    warnings.warn(f"MLflow logging failed: {exc}", stacklevel=2)
    finally:
        if parent_run_id is not None:
            end_mlflow_run_if_active_id(parent_run_id)

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
