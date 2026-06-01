"""Pool/worker entry points and the top-level OOD VQC sweep runner."""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from qml_project.circuit import VariationalClassifier, build_circuit
from qml_project.nim.data import training_subsets
from qml_project.parallel_sweep import run_pending_grid_tasks
from qml_project.training.mlflow_helpers import (
    _load_simulated_vqc_ood_from_mlflow,
    set_mlflow_tracking_uri,
)
from qml_project.training.results import SimulatedVQCSweepResults
from qml_project.training.types import (
    DecisionRule,
    LossName,
    MeasurementObservable,
    SimulatedVQCRunResult,
    VqcOodSweepTask,
)
from qml_project.training.vqc_factory import _make_vqc_factory

from .mlflow_io import _log_simulated_vqc_ood_result_to_mlflow
from .single_run import simulated_vqc_ood_single_run

_vqc_ood_pool: dict[str, Any] = {}


def _vqc_ood_pool_init(cfg: dict[str, Any]) -> None:
    """Worker initializer: one picklable dict (ProcessPoolExecutor initargs)."""
    _vqc_ood_pool.clear()
    _vqc_ood_pool.update(cfg)


def _vqc_ood_worker(task: VqcOodSweepTask) -> SimulatedVQCRunResult:
    p = _vqc_ood_pool
    ck = p["circuit_kwargs"]
    vc = build_circuit(**ck)
    return simulated_vqc_ood_single_run(
        vc,
        task.subset_X,
        task.subset_y,
        p["X_test"],
        p["y_test"],
        train_size=task.train_size,
        seed=task.seed,
        max_iter=p["max_iter"],
        shot_schedule=p["shot_schedule"],
        test_shots=p["test_shots"],
        sampler=None,
        decision_rule=p["decision_rule"],
        observable=p["observable"],
        loss_name=p["loss_name"],
        expectation_qubit=p["expectation_qubit"],
        feature_fn_for_policy=p["feature_fn_for_policy"],
        compute_win_rate=p["compute_win_rate"],
        n_games_win_rate=p["n_games_win_rate"],
        game_k=p["game_k"],
        game_M=p["game_M"],
        train_verbose=False,
        log_interval=p["log_interval"],
    )


def run_simulated_vqc_ood_sweep(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    vc_builder: Callable[[], VariationalClassifier] | None = None,
    circuit_kwargs: dict[str, Any] | None = None,
    train_sizes: Sequence[int | str] = (50, 100, "full"),
    seeds: Sequence[int] = tuple(range(10)),
    max_iter: int = 200,
    shot_schedule: dict[int, int] | None = None,
    test_shots: int = 300,
    sampler_factory: Callable[[int], Any] | None = None,
    decision_rule: DecisionRule = "argmax",
    observable: MeasurementObservable = "bitstring_probs",
    loss_name: LossName = "softmax_nll",
    expectation_qubit: int = 0,
    feature_fn_for_policy: Callable[[np.ndarray], np.ndarray] | None = None,
    compute_win_rate: bool = True,
    n_games_win_rate: int = 200,
    game_k: int = 3,
    game_M: int = 7,
    mlflow_experiment: str | None = None,
    mlflow_run_prefix: str = "simulated-vqc-ood",
    mlflow_extra_params: Mapping[str, Any] | None = None,
    use_cache: bool = True,
    run_pending: bool = True,
    verbose: bool = True,
    max_workers: int | None = None,
    log_interval: int = 20,
) -> SimulatedVQCSweepResults:
    """Run OOD VQC training at multiple train sizes and seeds.

    The caller should pass OOD arrays (train on M<=5, test on M>5) and encoded
    features. Train-size subsets are stratified per seed.

    When ``use_cache=True`` and ``mlflow_experiment`` is set, finished runs
    matching the grid (including ``mlflow_run_prefix`` in the run name) are
    loaded from MLflow; only missing points are trained. Each completed point is
    logged immediately so interrupted sweeps can resume. Set ``use_cache=False``
    to recompute the full grid (runs are still logged when an experiment is set).

    When ``run_pending=False``, grid cells that are **not** in the MLflow cache are
    skipped (no training). Cached cells are still returned. Use this to pull
    historical results for a branch of the grid without launching new work.

    Provide exactly one of ``vc_builder`` or ``circuit_kwargs``. For
    ``max_workers`` > 1, use ``circuit_kwargs`` (picklable) and omit
    ``sampler_factory``; MLflow is logged from the parent process.

    ``mlflow_extra_params`` (e.g. ``encoding``, ``config_id``, ``include_nim_sum``)
    is merged into each run's logged params as strings for the MLflow UI.
    """
    sweep_started = time.perf_counter()
    _mlflow_xp = dict(mlflow_extra_params) if mlflow_extra_params else {}
    if compute_win_rate and feature_fn_for_policy is None:
        raise ValueError(
            "feature_fn_for_policy is required when compute_win_rate=True."
        )

    if max_workers is not None and max_workers > 1:
        if circuit_kwargs is None:
            raise ValueError("circuit_kwargs is required when max_workers > 1.")
        if sampler_factory is not None:
            raise ValueError("sampler_factory is not supported when max_workers > 1.")

    factory = _make_vqc_factory(vc_builder, circuit_kwargs)

    int_sizes = [int(s) for s in train_sizes if isinstance(s, int)]
    mlflow_available = False
    if mlflow_experiment:
        try:
            import mlflow

            set_mlflow_tracking_uri()
            mlflow.set_experiment(mlflow_experiment)
            mlflow_available = True
        except ImportError:
            mlflow_available = False
            if verbose:
                print("Warning: MLflow not available; sweep runs will not be logged.")

    full_train_size = int(len(X_train))
    temp_vc = factory()
    vqc_cache: dict[tuple[int, int], SimulatedVQCRunResult] = {}
    if use_cache and mlflow_available and mlflow_experiment:
        vqc_cache = _load_simulated_vqc_ood_from_mlflow(
            mlflow_experiment,
            train_sizes,
            seeds,
            full_train_size=full_train_size,
            max_iter=max_iter,
            test_shots=test_shots,
            ansatz=str(temp_vc.ansatz),
            n_qubits=temp_vc.n_qubits,
            n_features=temp_vc.n_features,
            n_trainable=temp_vc.n_trainable,
            observable=observable,
            decision_rule=decision_rule,
            loss_name=loss_name,
            expectation_qubit=expectation_qubit,
            n_games_win_rate=n_games_win_rate,
            compute_win_rate=compute_win_rate,
            mlflow_run_prefix=mlflow_run_prefix,
        )
        if verbose and vqc_cache:
            print(f"  Loaded {len(vqc_cache)} simulated VQC runs from MLflow cache.")

    total_runs = len(seeds) * len(train_sizes)
    ordered: list[SimulatedVQCRunResult | None] = []
    pending: list[tuple[int, VqcOodSweepTask]] = []
    idx = 0
    run_idx = 0
    n_prefill_hits = 0
    n_skipped_uncached = 0

    for seed in seeds:
        per_seed_subsets = training_subsets(
            X_train,
            y_train,
            sizes=int_sizes,
            random_state=int(seed),
        )
        for tsz in train_sizes:
            if tsz == "full":
                subset = per_seed_subsets["full"]
            elif int(tsz) in per_seed_subsets:
                subset = per_seed_subsets[int(tsz)]
            else:
                continue

            run_idx += 1
            size = int(subset.size)
            ck = (size, int(seed))
            if ck in vqc_cache:
                ordered.append(vqc_cache[ck])
                n_prefill_hits += 1
                idx += 1
                if verbose and (max_workers is None or max_workers <= 1):
                    print(
                        f"[sim-vqc {run_idx}/{total_runs}] seed={seed} "
                        f"train_size={size} (cached)"
                    )
                continue

            if not run_pending:
                n_skipped_uncached += 1
                if verbose and (max_workers is None or max_workers <= 1):
                    print(
                        f"[sim-vqc {run_idx}/{total_runs}] seed={seed} "
                        f"train_size={size} (skipped — not in cache; run_pending=False)"
                    )
                continue

            if verbose and (max_workers is None or max_workers <= 1):
                print(
                    f"[sim-vqc {run_idx}/{total_runs}] seed={seed} train_size={size}"
                )

            task = VqcOodSweepTask(
                subset_X=np.asarray(subset.X, dtype=np.float64),
                subset_y=np.asarray(subset.y),
                seed=int(seed),
                train_size=size,
            )
            ordered.append(None)
            pending.append((idx, task))
            idx += 1

    def _log_ood(res: SimulatedVQCRunResult) -> None:
        if not mlflow_available:
            return
        _log_simulated_vqc_ood_result_to_mlflow(
            res,
            mlflow_run_prefix=mlflow_run_prefix,
            vc_meta=temp_vc,
            max_iter=max_iter,
            test_shots=test_shots,
            observable=observable,
            decision_rule=decision_rule,
            loss_name=loss_name,
            expectation_qubit=expectation_qubit,
            n_games_win_rate=n_games_win_rate,
            extra_params=_mlflow_xp,
        )

    if pending:

        def _run_serial(task: VqcOodSweepTask) -> SimulatedVQCRunResult:
            vc = factory()
            sampler = (
                sampler_factory(int(task.seed))
                if sampler_factory is not None
                else None
            )
            return simulated_vqc_ood_single_run(
                vc,
                task.subset_X,
                task.subset_y,
                X_test,
                y_test,
                train_size=task.train_size,
                seed=task.seed,
                max_iter=max_iter,
                shot_schedule=shot_schedule,
                test_shots=test_shots,
                sampler=sampler,
                decision_rule=decision_rule,
                observable=observable,
                loss_name=loss_name,
                expectation_qubit=expectation_qubit,
                feature_fn_for_policy=feature_fn_for_policy,
                compute_win_rate=compute_win_rate,
                n_games_win_rate=n_games_win_rate,
                game_k=game_k,
                game_M=game_M,
                train_verbose=verbose,
                log_interval=log_interval,
            )

        def _after_task(_j: int, res: SimulatedVQCRunResult) -> None:
            _log_ood(res)

        pool_cfg: dict[str, Any] | None = None
        if max_workers is not None and max_workers > 1:
            assert circuit_kwargs is not None
            pool_cfg = {
                "circuit_kwargs": dict(circuit_kwargs),
                "X_test": X_test,
                "y_test": y_test,
                "max_iter": max_iter,
                "shot_schedule": shot_schedule,
                "test_shots": test_shots,
                "decision_rule": decision_rule,
                "observable": observable,
                "loss_name": loss_name,
                "expectation_qubit": expectation_qubit,
                "feature_fn_for_policy": feature_fn_for_policy,
                "compute_win_rate": compute_win_rate,
                "n_games_win_rate": n_games_win_rate,
                "game_k": game_k,
                "game_M": game_M,
                "log_interval": log_interval,
            }

        run_pending_grid_tasks(
            pending,
            ordered,
            run_serial=_run_serial,
            parallel_worker=_vqc_ood_worker,
            max_workers=max_workers,
            tqdm_desc=mlflow_run_prefix,
            initializer=_vqc_ood_pool_init if pool_cfg is not None else None,
            initargs=(pool_cfg,) if pool_cfg is not None else (),
            on_task_complete=_after_task,
        )

    sweep = SimulatedVQCSweepResults()
    sweep.results = [r for r in ordered if r is not None]
    sweep.sweep_metadata = {
        "elapsed_wall_time_s": float(time.perf_counter() - sweep_started),
        "max_workers": (
            int(max_workers)
            if (max_workers is not None and max_workers > 1)
            else 1
        ),
        "n_tasks": int(total_runs),
        "n_cached": int(n_prefill_hits),
        "n_skipped_uncached": int(n_skipped_uncached),
        "run_pending": bool(run_pending),
    }
    return sweep


__all__ = ["run_simulated_vqc_ood_sweep"]
