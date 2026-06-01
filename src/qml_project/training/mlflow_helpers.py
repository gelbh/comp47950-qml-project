"""MLflow tracking URI and cache loaders for training sweeps."""

from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

import numpy as np

from qml_project.training.experiment_namespace import resolve_experiment
from qml_project.training.mlflow_run_index import run_row_from_mlflow
from qml_project.training.mlflow_schema import MetricKey, ParamKey, PipelineValue, RegimeValue
from qml_project.training.selection import Winner
from qml_project.training.types import (
    DecisionRule,
    ExperimentResult,
    LossName,
    MeasurementObservable,
    MultiSeedSummary,
    SimulatedVQCRunResult,
    TrainingHistory,
)


def set_mlflow_tracking_uri() -> None:
    """Point MLflow at ``<repo>/mlruns`` unless ``MLFLOW_TRACKING_URI`` is already set."""
    _root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
    )
    os.environ.setdefault(
        "MLFLOW_TRACKING_URI", os.path.join(_root, "mlruns")
    )


def stringify_mlflow_param(value: Any) -> str:
    """Normalise values for ``mlflow.log_params`` (UI grouping / compare columns)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def stringify_mlflow_params(params: Mapping[str, Any]) -> dict[str, str]:
    """Return a flat string dict safe for ``mlflow.log_params``."""
    return {str(k): stringify_mlflow_param(v) for k, v in params.items()}


def parse_mlflow_bool(value: str | None, *, default: bool = True) -> bool:
    """Parse a stringified MLflow boolean param back to ``bool``.

    Inverse of :func:`stringify_mlflow_param` for boolean values. Accepts the
    canonical ``"true"`` / ``"false"`` plus a few common variants
    (``"1"`` / ``"0"``, ``"yes"`` / ``"no"``). Unrecognised values return *default*.
    """
    if value is None:
        return default
    s = str(value).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return default


def log_training_run(
    mlflow: Any,
    *,
    run_name: str,
    params: Mapping[str, Any],
    metrics: Mapping[str, float],
    tags: Mapping[str, str] | None = None,
) -> None:
    """Start one run, log stringified params, optional tags, then metrics.

    Sweep resume loaders match on **params** (via :func:`stringify_mlflow_params`);
    **tags** are for MLflow UI filters and grouping (see ``standard_tags``).
    """
    str_params = stringify_mlflow_params(params)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(str_params)
        if tags:
            for key, val in tags.items():
                mlflow.set_tag(str(key), str(val))
        mlflow.log_metrics(dict(metrics))


def end_mlflow_run_if_nested_under(
    parent_run_id: str,
    *,
    status: str = "FAILED",
) -> None:
    """End the active MLflow run if it is a nested child of *parent_run_id*.

    Used on ``KeyboardInterrupt`` or other aborts so nested runs started under
    a multi-seed parent do not stay open in the tracking client.
    """
    try:
        import mlflow
    except ImportError:
        return
    ar = mlflow.active_run()
    if ar is None:
        return
    pid = ar.data.tags.get("mlflow.parentRunId")
    if pid == parent_run_id:
        mlflow.end_run(status=status)


def end_mlflow_run_if_active_id(
    run_id: str,
    *,
    status: str = "FAILED",
) -> None:
    """End the active MLflow run if its id equals *run_id*."""
    try:
        import mlflow
    except ImportError:
        return
    ar = mlflow.active_run()
    if ar is None:
        return
    if ar.info.run_id == run_id:
        mlflow.end_run(status=status)


def end_all_active_mlflow_runs(*, status: str = "FAILED") -> int:
    """Pop every active run from the client stack (innermost first).

    For notebook recovery after a bad interrupt. Avoid calling this while you
    intentionally rely on nested runs from unrelated code.

    Returns
    -------
    int
        Number of runs closed.
    """
    try:
        import mlflow
    except ImportError:
        return 0
    n = 0
    while mlflow.active_run() is not None:
        mlflow.end_run(status=status)
        n += 1
    return n


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


def _merge_multi_seed_finished_child_runs(
    client: Any,
    experiment_id: str,
    mlflow_run_name: str,
    wanted_params: dict[str, str],
    *,
    max_children_per_parent: int = 128,
) -> dict[int, Any]:
    """Collect FINISHED nested seed runs across all matching parent attempts.

    Parents are matched by *run_name* and param signature (any lifecycle status).
    Iterating parents **newest first**, the first FINISHED child seen for each
    seed wins, so resumed runs prefer the latest parent that logged that seed.
    """
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["start_time DESC"],
        max_results=500,
    )
    parents: list[Any] = []
    for run in runs:
        if run.data.tags.get("mlflow.parentRunId"):
            continue
        name = run.info.run_name or run.data.tags.get("mlflow.runName") or ""
        if name != mlflow_run_name:
            continue
        if not _params_match_mlflow(run.data.params, wanted_params):
            continue
        parents.append(run)
    parents.sort(key=lambda r: r.info.start_time or 0, reverse=True)

    by_seed: dict[int, Any] = {}
    for p in parents:
        filt = f"tags.mlflow.parentRunId = '{p.info.run_id}'"
        child_runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=filt,
            max_results=max_children_per_parent,
        )
        for cr in child_runs:
            if cr.info.status != "FINISHED":
                continue
            sp = cr.data.params.get("seed")
            if sp is None:
                continue
            try:
                si = int(sp)
            except (TypeError, ValueError):
                continue
            if si not in by_seed:
                by_seed[si] = cr
    return by_seed


def _experiment_result_from_mlflow_child_run(
    child_run: Any,
    *,
    seed: int,
    n_classes: int,
) -> ExperimentResult | None:
    metrics = child_run.data.metrics
    if "balanced_accuracy" not in metrics or "mcc" not in metrics:
        return None
    hist = TrainingHistory(
        best_loss=float(metrics.get("final_loss", 0.0)),
        total_training_time=float(metrics.get("training_time", 0.0)),
        total_evals=0,
    )
    return ExperimentResult(
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


def _partial_multi_seed_cached_experiment_results(
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
) -> dict[int, ExperimentResult]:
    """Map seed -> cached :class:`ExperimentResult` from any prior parent attempt."""
    try:
        from mlflow.tracking import MlflowClient
    except ImportError:
        return {}

    set_mlflow_tracking_uri()
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        return {}

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
    merge = _merge_multi_seed_finished_child_runs(
        client, exp.experiment_id, mlflow_run_name, wanted
    )
    out: dict[int, ExperimentResult] = {}
    for s in seeds:
        if s not in merge:
            continue
        er = _experiment_result_from_mlflow_child_run(
            merge[s], seed=s, n_classes=n_classes
        )
        if er is not None:
            out[s] = er
    return out


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
    """Restore :class:`MultiSeedSummary` from nested runs (any parent lifecycle).

    Merges FINISHED children across all parent runs with the same *run_name* and
    param signature, so a completed grid can load even if an older parent ended
    in FAILED after a later parent finished successfully.
    """
    try:
        from mlflow.tracking import MlflowClient
    except ImportError:
        return None

    set_mlflow_tracking_uri()
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
    by_seed = _merge_multi_seed_finished_child_runs(
        client, exp.experiment_id, mlflow_run_name, wanted
    )
    if set(by_seed.keys()) != set(seeds):
        return None

    all_results: list[ExperimentResult] = []
    for seed in seeds:
        er = _experiment_result_from_mlflow_child_run(
            by_seed[seed], seed=seed, n_classes=n_classes
        )
        if er is None:
            return None
        all_results.append(er)

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

    set_mlflow_tracking_uri()
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
        if p.get(ParamKey.PIPELINE) != PipelineValue.SIMULATED_VQC or p.get(ParamKey.REGIME) != RegimeValue.OOD:
            return False
        checks = [
            (ParamKey.MAX_ITER, str(max_iter)),
            (ParamKey.TEST_SHOTS, str(test_shots)),
            (ParamKey.ANSATZ, ansatz),
            (ParamKey.N_QUBITS, str(n_qubits)),
            (ParamKey.N_FEATURES, str(n_features)),
            (ParamKey.N_TRAINABLE, str(n_trainable)),
            (ParamKey.OBSERVABLE, observable),
            (ParamKey.DECISION_RULE, decision_rule),
            (ParamKey.LOSS_NAME, loss_name),
            (ParamKey.EXPECTATION_QUBIT, str(expectation_qubit)),
            (ParamKey.N_GAMES_WIN_RATE, str(n_games_win_rate)),
        ]
        for k, v in checks:
            if p.get(k) != v:
                return False
        return True

    cache: dict[tuple[int, int], SimulatedVQCRunResult] = {}

    for run in runs:
        row = run_row_from_mlflow(run)
        if row.status != "FINISHED":
            continue
        p = row.params
        m = row.metrics
        if not _params_match(p):
            continue
        try:
            train_size_int = int(p[ParamKey.TRAIN_SIZE])
            seed_int = int(p[ParamKey.SEED])
        except (KeyError, TypeError, ValueError):
            continue
        key = (train_size_int, seed_int)
        if key not in wanted or key in cache:
            continue
        expected_name = f"{mlflow_run_prefix}|n={train_size_int}|s={seed_int}"
        if row.run_name != expected_name:
            continue
        if compute_win_rate and MetricKey.WIN_RATE not in m:
            continue

        wr: float | None = (
            float(m[MetricKey.WIN_RATE]) if MetricKey.WIN_RATE in m else None
        )
        cache[key] = SimulatedVQCRunResult(
            train_size=train_size_int,
            seed=seed_int,
            test_accuracy=float(m.get(MetricKey.TEST_ACCURACY, 0.0)),
            balanced_accuracy=float(m.get(MetricKey.BALANCED_ACCURACY, 0.0)),
            mcc=float(m.get(MetricKey.MCC, 0.0)),
            win_rate=wr,
            training_time=float(m.get(MetricKey.TRAINING_TIME, 0.0)),
            inference_time=float(m.get(MetricKey.INFERENCE_TIME, 0.0)),
            final_loss=float(m.get(MetricKey.FINAL_LOSS, 0.0)),
            ansatz=ansatz,
            observable=observable,  # type: ignore[arg-type]
            decision_rule=decision_rule,  # type: ignore[arg-type]
            loss_name=loss_name,  # type: ignore[arg-type]
        )

    return cache


def _log_one_selection_winner_run(mlflow: Any, w: Winner, *, scope: str) -> None:
    try:
        with mlflow.start_run(
            run_name=f"selection|{scope}|{w.pipeline}|{w.config_id}"
        ):
            mlflow.set_tags(
                {
                    "pipeline": "selection",
                    "stage": "selection",
                    "winner_scope": scope,
                    "winner_pipeline": w.pipeline,
                    "winner_config_id": w.config_id,
                    "winner_encoding": w.encoding or "",
                }
            )
            metrics: dict[str, float] = {
                "winner_mean_accuracy": float(w.mean_accuracy),
                "winner_std_accuracy": float(w.std_accuracy),
            }
            if w.mean_cost is not None:
                metrics["winner_mean_cost"] = float(w.mean_cost)
            if w.train_size_used is not None:
                metrics["winner_train_size_used"] = float(w.train_size_used)
            mlflow.log_metrics(metrics)
            mlflow.log_param("rationale", w.rationale)
    except Exception as exc:
        print(f"(MLflow selection log skipped for {scope}/{w.pipeline}: {exc})")


def log_quantum_selection_winners_to_mlflow(
    quantum_winners: Mapping[str, Winner],
    overall_winner: Winner,
    *,
    mlflow_experiment: str | None = None,
) -> None:
    """Log Section 07 quantum selection winners to MLflow (one run per scope).

    Uses ``nim.<pipeline>.<stage>`` experiment
    ``resolve_experiment("selection", "selection")`` unless *mlflow_experiment*
    is set. No-op if ``mlflow`` is not installed.
    """
    try:
        import mlflow
    except ImportError:
        return
    experiment_name = mlflow_experiment or resolve_experiment("selection", "selection")
    try:
        set_mlflow_tracking_uri()
        mlflow.set_experiment(experiment_name)
    except Exception as exc:
        print(f"(MLflow selection log skipped: {exc})")
        return
    for scope, w in quantum_winners.items():
        _log_one_selection_winner_run(mlflow, w, scope=scope)
    _log_one_selection_winner_run(mlflow, overall_winner, scope="overall")
