"""Top-level classical sweep orchestrator with optional parallel dispatch."""

from __future__ import annotations

import time
import warnings
from typing import Sequence

import numpy as np

from qml_project.baselines.evaluation import ClassicalResult
from qml_project.baselines.sweep_results import SweepResults
from qml_project.nim.data import augment_s3, canonical_order, training_subsets
from qml_project.parallel_sweep import run_pending_grid_tasks
from qml_project.training.mlflow_helpers import set_mlflow_tracking_uri

from .cache import load_classical_sweep_cache, log_classical_mlflow_run
from .tasks import (
    ClassicalSweepTask,
    classical_sweep_pool_init,
    classical_sweep_worker,
    execute_classical_sweep_task,
)


def run_classical_sweep(
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    *,
    model_names: Sequence[str],
    feature_sets: Sequence[str],
    symmetry_variants: Sequence[str],
    train_sizes: Sequence[int | str],
    seeds: Sequence[int],
    M: int,
    compute_win_rate: bool,
    n_games_win_rate: int,
    mlflow_experiment: str,
    use_cache: bool,
    max_workers: int | None,
    c_svc: float = 1.0,
) -> SweepResults:
    """Enumerate the classical baseline grid, optionally resume from MLflow, run tasks."""
    sweep_started = time.perf_counter()
    c_svc_f = float(c_svc)

    mlflow_mod = None
    try:
        import mlflow as _mlflow

        set_mlflow_tracking_uri()
        _mlflow.set_experiment(mlflow_experiment)
        mlflow_mod = _mlflow
    except ImportError:
        warnings.warn("MLflow not installed; skipping logging.", stacklevel=2)

    cache: dict[tuple[str, str, str, int, int, float], ClassicalResult] = {}
    if use_cache and mlflow_mod is not None:
        cache = load_classical_sweep_cache(
            mlflow_experiment,
            model_names,
            feature_sets,
            symmetry_variants,
            train_sizes,
            seeds,
            "ood",
            full_train_size=len(X_train_raw),
            c_svc=c_svc_f,
        )
        if cache:
            print(f"  Loaded {len(cache)} runs from MLflow cache.")

    total = (
        len(model_names)
        * len(feature_sets)
        * len(symmetry_variants)
        * len(train_sizes)
        * len(seeds)
    )
    ordered: list[ClassicalResult | None] = []
    pending: list[tuple[int, ClassicalSweepTask]] = []
    idx = 0

    for model_name in model_names:
        for fs in feature_sets:
            for sym in symmetry_variants:
                for seed in seeds:
                    int_sizes = [s for s in train_sizes if isinstance(s, int)]
                    seed_subsets = training_subsets(
                        X_train_raw,
                        y_train,
                        sizes=int_sizes,
                        random_state=seed,
                    )
                    for tsz in train_sizes:
                        if tsz == "full":
                            X_sub, y_sub = X_train_raw, y_train
                        elif tsz in seed_subsets:
                            X_sub = seed_subsets[tsz].X
                            y_sub = seed_subsets[tsz].y
                        else:
                            continue

                        result_train_size = (
                            len(X_sub) if tsz == "full" else int(tsz)
                        )
                        cache_key = (
                            model_name,
                            fs,
                            sym,
                            result_train_size,
                            int(seed),
                            c_svc_f,
                        )
                        if cache_key in cache:
                            ordered.append(cache[cache_key])
                            idx += 1
                            continue

                        if sym == "augmented":
                            X_sub_use, y_sub_use = augment_s3(
                                X_sub, y_sub, deduplicate=True
                            )
                        elif sym == "canonical":
                            X_sub_use, _ = canonical_order(
                                np.asarray(X_sub, dtype=np.int32)
                            )
                            y_sub_use = y_sub
                        elif sym == "none":
                            X_sub_use, y_sub_use = X_sub, y_sub
                        else:
                            raise ValueError(f"Unknown symmetry variant: {sym!r}")

                        train_size_val = tsz if isinstance(tsz, int) else len(X_sub)
                        task = ClassicalSweepTask(
                            X_sub_raw=np.asarray(X_sub_use, dtype=np.int32),
                            y_sub=np.asarray(y_sub_use),
                            model_name=model_name,
                            feature_set=str(fs),
                            symmetry=sym,
                            train_size=int(train_size_val),
                            seed=int(seed),
                            compute_win_rate=compute_win_rate,
                            n_games_win_rate=n_games_win_rate,
                            c_svc=c_svc_f,
                        )
                        ordered.append(None)
                        pending.append((idx, task))
                        idx += 1

    if pending:

        def _run_serial(t: ClassicalSweepTask) -> ClassicalResult:
            return execute_classical_sweep_task(t, X_test_raw, y_test, M)

        def _after_task(j: int, res: ClassicalResult) -> None:
            if mlflow_mod is not None:
                log_classical_mlflow_run(res, mlflow_mod)

        run_pending_grid_tasks(
            pending,
            ordered,
            run_serial=_run_serial,
            parallel_worker=classical_sweep_worker,
            max_workers=max_workers,
            tqdm_desc=mlflow_experiment,
            initializer=classical_sweep_pool_init,
            initargs=(X_test_raw, y_test, M),
            on_task_complete=_after_task,
        )

    bundle = SweepResults()
    bundle.results = [r for r in ordered if r is not None]
    bundle.sweep_metadata = {
        "elapsed_wall_time_s": float(time.perf_counter() - sweep_started),
        "max_workers": (
            int(max_workers) if (max_workers is not None and max_workers > 1) else 1
        ),
        "n_tasks": int(total),
        "n_cached": int(total - len(pending)),
    }
    print(f"  Sweep complete: {len(bundle.results)}/{total} runs.")
    return bundle
