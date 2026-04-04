"""MLflow tracking URI and cache loaders for training sweeps."""

from __future__ import annotations

import os
from typing import Any, Sequence

import numpy as np

from qml_project.training.types import (
    DecisionRule,
    ExperimentResult,
    LossName,
    MeasurementObservable,
    MultiSeedSummary,
    SimulatedVQCRunResult,
    TrainingHistory,
)


def _set_mlflow_tracking_uri() -> None:
    _root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    )
    os.environ.setdefault(
        "MLFLOW_TRACKING_URI", os.path.join(_root, "mlruns")
    )


def _parent_run_param_signature(
    *,
    seeds: list[int],
    max_iter: int,
    test_shots: int,
    n_qubits: int,
    n_features: int,
    n_classes: int,
    n_trainable: int,
    ansatz: str,
    observable: str,
    decision_rule: str,
    loss_name: str,
    expectation_qubit: int,
) -> dict[str, str]:
    return {
        "n_seeds": str(len(seeds)),
        "seeds": ",".join(str(s) for s in seeds),
        "max_iter": str(max_iter),
        "test_shots": str(test_shots),
        "n_qubits": str(n_qubits),
        "n_features": str(n_features),
        "n_classes": str(n_classes),
        "n_trainable": str(n_trainable),
        "ansatz": str(ansatz),
        "observable": str(observable),
        "decision_rule": str(decision_rule),
        "loss_name": str(loss_name),
        "expectation_qubit": str(expectation_qubit),
    }


def _params_match_mlflow(stored: dict[str, str], wanted: dict[str, str]) -> bool:
    return all(stored.get(k) == v for k, v in wanted.items())


def _load_multi_seed_summary_from_mlflow(
    experiment_name: str,
    mlflow_run_name: str,
    *,
    seeds: list[int],
    max_iter: int,
    test_shots: int,
    n_qubits: int,
    n_features: int,
    n_classes: int,
    n_trainable: int,
    ansatz: str,
    observable: MeasurementObservable,
    decision_rule: DecisionRule,
    loss_name: LossName,
    expectation_qubit: int,
    verbose: bool,
) -> MultiSeedSummary | None:
    """Restore :class:`MultiSeedSummary` from a logged parent + nested runs."""
    try:
        from mlflow.tracking import MlflowClient
    except ImportError:
        return None

    _set_mlflow_tracking_uri()
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        return None

    wanted = _parent_run_param_signature(
        seeds=seeds,
        max_iter=max_iter,
        test_shots=test_shots,
        n_qubits=n_qubits,
        n_features=n_features,
        n_classes=n_classes,
        n_trainable=n_trainable,
        ansatz=ansatz,
        observable=observable,
        decision_rule=decision_rule,
        loss_name=loss_name,
        expectation_qubit=expectation_qubit,
    )

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["end_time DESC"],
        max_results=500,
    )

    parent = None
    for run in runs:
        if run.info.status != "FINISHED":
            continue
        if run.data.tags.get("mlflow.parentRunId"):
            continue
        name = run.info.run_name or run.data.tags.get("mlflow.runName") or ""
        if name != mlflow_run_name:
            continue
        if not _params_match_mlflow(run.data.params, wanted):
            continue
        parent = run
        break

    if parent is None:
        return None

    filt = f"tags.mlflow.parentRunId = '{parent.info.run_id}'"
    child_runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=filt,
        max_results=max(len(seeds) * 4, 32),
    )
    finished_children = [cr for cr in child_runs if cr.info.status == "FINISHED"]
    finished_children.sort(
        key=lambda r: r.info.end_time or 0,
        reverse=True,
    )
    by_seed: dict[int, Any] = {}
    for cr in finished_children:
        sp = cr.data.params.get("seed")
        if sp is None:
            continue
        try:
            si = int(sp)
        except (TypeError, ValueError):
            continue
        if si in by_seed:
            continue
        by_seed[si] = cr

    if set(by_seed.keys()) != set(seeds):
        return None

    all_results: list[ExperimentResult] = []
    for seed in seeds:
        child_run = by_seed[seed]
        metrics = child_run.data.metrics
        if "balanced_accuracy" not in metrics or "mcc" not in metrics:
            return None
        hist = TrainingHistory(
            best_loss=float(metrics.get("final_loss", 0.0)),
            total_training_time=float(metrics.get("training_time", 0.0)),
            total_evals=0,
        )
        all_results.append(
            ExperimentResult(
                seed=seed,
                best_weights=np.array([], dtype=np.float64),
                history=hist,
                test_accuracy=float(metrics.get("test_accuracy", 0.0)),
                test_predictions=np.array([], dtype=np.int64),
                test_class_probs=np.zeros((0, n_classes), dtype=np.float64),
                training_time=float(metrics.get("training_time", 0.0)),
                inference_time=float(metrics.get("inference_time", 0.0)),
                balanced_accuracy=float(metrics["balanced_accuracy"]),
                mcc=float(metrics["mcc"]),
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
    if verbose:
        print(
            f"  Loaded multi-seed summary from MLflow ({experiment_name!r}, "
            f"run_name={mlflow_run_name!r})."
        )
    return summary


def _load_simulated_vqc_ood_from_mlflow(
    experiment_name: str,
    train_sizes: Sequence[int | str],
    seeds: Sequence[int],
    *,
    full_train_size: int,
    max_iter: int,
    test_shots: int,
    ansatz: str,
    n_qubits: int,
    n_features: int,
    n_trainable: int,
    observable: str,
    decision_rule: str,
    loss_name: str,
    expectation_qubit: int,
    n_games_win_rate: int,
    compute_win_rate: bool,
    mlflow_run_prefix: str,
) -> dict[tuple[int, int], SimulatedVQCRunResult]:
    """Load simulated VQC OOD sweep runs from MLflow (newest run wins per key)."""
    try:
        from mlflow.tracking import MlflowClient
    except ImportError:
        return {}

    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        return {}

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["end_time DESC"],
        max_results=10_000,
    )

    wanted: set[tuple[int, int]] = set()
    for tsz in train_sizes:
        size = full_train_size if tsz == "full" else int(tsz)
        for seed in seeds:
            wanted.add((size, int(seed)))

    def _params_match(p: dict[str, str]) -> bool:
        if p.get("pipeline") != "simulated_vqc" or p.get("regime") != "ood":
            return False
        checks = [
            ("max_iter", str(max_iter)),
            ("test_shots", str(test_shots)),
            ("ansatz", ansatz),
            ("n_qubits", str(n_qubits)),
            ("n_features", str(n_features)),
            ("n_trainable", str(n_trainable)),
            ("observable", observable),
            ("decision_rule", decision_rule),
            ("loss_name", loss_name),
            ("expectation_qubit", str(expectation_qubit)),
            ("n_games_win_rate", str(n_games_win_rate)),
        ]
        for k, v in checks:
            if p.get(k) != v:
                return False
        return True

    cache: dict[tuple[int, int], SimulatedVQCRunResult] = {}

    for run in runs:
        if run.info.status != "FINISHED":
            continue
        p = run.data.params
        m = run.data.metrics
        if not _params_match(p):
            continue
        try:
            train_size_int = int(p["train_size"])
            seed_int = int(p["seed"])
        except (KeyError, TypeError, ValueError):
            continue
        key = (train_size_int, seed_int)
        if key not in wanted or key in cache:
            continue
        expected_name = f"{mlflow_run_prefix}|n={train_size_int}|s={seed_int}"
        if run.info.run_name != expected_name:
            continue
        if compute_win_rate and "win_rate" not in m:
            continue

        wr: float | None = float(m["win_rate"]) if "win_rate" in m else None
        cache[key] = SimulatedVQCRunResult(
            train_size=train_size_int,
            seed=seed_int,
            test_accuracy=float(m.get("test_accuracy", 0.0)),
            balanced_accuracy=float(m.get("balanced_accuracy", 0.0)),
            mcc=float(m.get("mcc", 0.0)),
            win_rate=wr,
            training_time=float(m.get("training_time", 0.0)),
            inference_time=float(m.get("inference_time", 0.0)),
            final_loss=float(m.get("final_loss", 0.0)),
            ansatz=ansatz,
            observable=observable,  # type: ignore[arg-type]
            decision_rule=decision_rule,  # type: ignore[arg-type]
            loss_name=loss_name,  # type: ignore[arg-type]
        )

    return cache
