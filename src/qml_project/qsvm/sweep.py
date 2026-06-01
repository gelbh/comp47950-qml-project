"""QSVM sample-efficiency sweep with optional multiprocessing.

Pool init / worker entry points are defined at module level so the spawn
multiprocessing context can pickle them by fully-qualified name.
"""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, cast

import numpy as np

from qml_project.nim.data import training_subsets
from qml_project.nim.encoding import EncodingName, SymmetryMode
from qml_project.parallel_sweep import run_pending_grid_tasks
from qml_project.training.mlflow_helpers import set_mlflow_tracking_uri

from .kernel import KernelBackend, KernelEstimatorMode
from .mlflow_io import _load_qsvm_sweep_from_mlflow, _log_mlflow_qsvm
from .model import (
    QuantumKernelResult,
    QuantumKernelSweepResults,
    evaluate_quantum_kernel_svm,
    evaluate_qsvm_win_rate,
    fit_quantum_kernel_svm,
)

_LOG = logging.getLogger(__name__)
_qsvm_pool: dict[str, Any] = {}


def _normalise_include_nim_sum_values(
    include_nim_sum: bool | Sequence[bool],
) -> tuple[bool, ...]:
    """Expand sweep flag to a tuple (single bool is wrapped as a one-tuple)."""
    if isinstance(include_nim_sum, bool):
        return (include_nim_sum,)
    return tuple(bool(x) for x in include_nim_sum)


def _setup_logging_for_verbose(verbose: bool) -> None:
    """Ensure INFO logs are visible in notebook/CLI when verbose=True."""
    if not verbose:
        return
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    if _LOG.level > logging.INFO:
        _LOG.setLevel(logging.INFO)


@dataclass(frozen=True)
class QsvmSweepTask:
    """One QSVM grid point (materialised train subset)."""

    encoding: str
    seed: int
    X_train_raw: np.ndarray
    y_train: np.ndarray
    train_size: int
    include_nim_sum: bool
    c_svc: float
    estimator_mode: KernelEstimatorMode
    kernel_backend: KernelBackend
    shots: int


def _qsvm_sweep_pool_init(
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    class_weight: str | dict[int, float] | None,
    M: int,
    bits_per_heap: int,
    iqp_reps: int,
    symmetry: SymmetryMode,
    compute_win_rate: bool,
    n_games_win_rate: int,
    c_svc: float,
    estimator_mode: KernelEstimatorMode,
    kernel_backend: KernelBackend,
    shots: int,
) -> None:
    _qsvm_pool.clear()
    _qsvm_pool["X_test_raw"] = X_test_raw
    _qsvm_pool["y_test"] = y_test
    _qsvm_pool["class_weight"] = class_weight
    _qsvm_pool["M"] = M
    _qsvm_pool["bits_per_heap"] = bits_per_heap
    _qsvm_pool["iqp_reps"] = iqp_reps
    _qsvm_pool["symmetry"] = symmetry
    _qsvm_pool["compute_win_rate"] = compute_win_rate
    _qsvm_pool["n_games_win_rate"] = n_games_win_rate
    _qsvm_pool["c_svc"] = c_svc
    _qsvm_pool["estimator_mode"] = estimator_mode
    _qsvm_pool["kernel_backend"] = kernel_backend
    _qsvm_pool["shots"] = shots


def execute_qsvm_sweep_task(
    task: QsvmSweepTask,
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    *,
    class_weight: str | dict[int, float] | None,
    M: int,
    bits_per_heap: int,
    iqp_reps: int,
    include_nim_sum: bool,
    symmetry: SymmetryMode,
    compute_win_rate: bool,
    n_games_win_rate: int,
    c_svc: float,
    estimator_mode: KernelEstimatorMode,
    kernel_backend: KernelBackend,
    shots: int,
) -> QuantumKernelResult:
    """Fit and evaluate QSVM for one grid point."""
    encoding = cast(EncodingName, task.encoding)
    qsvm, train_time, kernel_matrix_time_s = fit_quantum_kernel_svm(
        task.X_train_raw,
        task.y_train,
        encoding=encoding,
        class_weight=class_weight,
        M=M,
        bits_per_heap=bits_per_heap,
        iqp_reps=iqp_reps,
        include_nim_sum=include_nim_sum,
        symmetry=symmetry,
        random_state=int(task.seed),
        c_svc=float(c_svc),
        estimator_mode=estimator_mode,
        kernel_backend=kernel_backend,
        shots=int(shots),
    )
    res, _ = evaluate_quantum_kernel_svm(qsvm, X_test_raw, y_test)
    res.seed = int(task.seed)
    res.train_size = int(task.train_size)
    res.train_time_s = float(train_time)
    res.kernel_matrix_time_s = float(kernel_matrix_time_s)
    res.include_nim_sum = bool(include_nim_sum)
    if compute_win_rate:
        res.win_rate = evaluate_qsvm_win_rate(
            qsvm,
            n_games=n_games_win_rate,
            k=3,
            M=M,
            seed=int(task.seed),
        )
    return res


def _qsvm_sweep_worker(task: QsvmSweepTask) -> QuantumKernelResult:
    return execute_qsvm_sweep_task(
        task,
        _qsvm_pool["X_test_raw"],
        _qsvm_pool["y_test"],
        class_weight=_qsvm_pool["class_weight"],
        M=_qsvm_pool["M"],
        bits_per_heap=_qsvm_pool["bits_per_heap"],
        iqp_reps=_qsvm_pool["iqp_reps"],
        include_nim_sum=task.include_nim_sum,
        symmetry=_qsvm_pool["symmetry"],
        compute_win_rate=_qsvm_pool["compute_win_rate"],
        n_games_win_rate=_qsvm_pool["n_games_win_rate"],
        c_svc=_qsvm_pool["c_svc"],
        estimator_mode=_qsvm_pool["estimator_mode"],
        shots=_qsvm_pool["shots"],
        kernel_backend=_qsvm_pool["kernel_backend"],
    )


def run_quantum_kernel_sweep(
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    *,
    encodings: Sequence[EncodingName] = ("angle", "amplitude", "binary"),
    train_sizes: Sequence[int | str] = (50, 100, "full"),
    seeds: Sequence[int] = tuple(range(10)),
    class_weight: str | dict[int, float] | None = "balanced",
    M: int = 7,
    bits_per_heap: int = 3,
    iqp_reps: int = 2,
    include_nim_sum: bool | Sequence[bool] = True,
    symmetry: SymmetryMode = "none",
    compute_win_rate: bool = True,
    n_games_win_rate: int = 200,
    mlflow_experiment: str | None = None,
    mlflow_run_prefix: str | None = None,
    mlflow_extra_params: Mapping[str, Any] | None = None,
    use_cache: bool = True,
    verbose: bool = True,
    max_workers: int | None = None,
    c_svc: float = 1.0,
    estimator_mode: KernelEstimatorMode = "exact_statevector",
    kernel_backend: KernelBackend = "manual",
    shots: int = 1024,
) -> QuantumKernelSweepResults:
    """Run multi-seed QSVM sample-efficiency sweep (OOD-ready).

    When ``use_cache=True`` and ``mlflow_experiment`` is set, finished runs
    in that experiment matching the sweep grid **and**
    ``encoding_cache_revision`` (must equal ``QSVM_ENCODING_CACHE_REVISION``) are
    loaded from MLflow; only missing grid points are computed. Older runs
    without that param or with a different revision are ignored so encoding
    changes do not resurrect stale metrics. Set ``use_cache=False`` to skip
    MLflow resume entirely, or bump ``QSVM_ENCODING_CACHE_REVISION`` in
    ``qsvm/mlflow_io.py`` when the feature map definition changes again.

    When ``max_workers`` > 1, each grid point runs in a subprocess; MLflow
    logging happens in the parent as each run finishes. ``class_weight`` must be picklable (e.g.
    ``\"balanced\"`` or ``None``); dict weights are supported if hashable
    via pickling.

    ``include_nim_sum`` may be a single bool or a sequence (e.g. ``(True, False)``)
    to ablate amplitude-style encodings while reusing the same train subsets.

    ``mlflow_extra_params`` is merged into each logged run (stringified) so the
    MLflow UI can filter/group on keys beyond the built-in sweep columns.
    """
    sweep_started = time.perf_counter()
    _setup_logging_for_verbose(verbose)
    effective_tqdm_desc = mlflow_run_prefix or mlflow_experiment or "QSVM sweep"
    inc_grid = _normalise_include_nim_sum_values(include_nim_sum)
    _mlflow_xp: dict[str, Any] = dict(mlflow_extra_params) if mlflow_extra_params else {}
    mlflow_mod = None
    if mlflow_experiment:
        try:
            import mlflow as _mlflow

            set_mlflow_tracking_uri()
            _mlflow.set_experiment(mlflow_experiment)
            mlflow_mod = _mlflow
        except ImportError:
            warnings.warn("MLflow not installed; skipping logging.", stacklevel=2)

    full_train_size = int(len(X_train_raw))
    cache: dict[tuple[str, int, int, bool], QuantumKernelResult] = {}
    if use_cache and mlflow_mod is not None and mlflow_experiment:
        cache = _load_qsvm_sweep_from_mlflow(
            mlflow_experiment,
            encodings,
            train_sizes,
            seeds,
            full_train_size=full_train_size,
            symmetry=symmetry,
            include_nim_sum_values=inc_grid,
            bits_per_heap=bits_per_heap,
            iqp_reps=iqp_reps,
            compute_win_rate=compute_win_rate,
            c_svc=float(c_svc),
            estimator_mode=estimator_mode,
            kernel_backend=kernel_backend,
            shots=int(shots) if estimator_mode == "shot_binomial" else None,
        )
        if verbose and cache:
            _LOG.info("  Loaded %d QSVM runs from MLflow cache.", len(cache))

    int_sizes = [s for s in train_sizes if isinstance(s, int)]
    total = len(encodings) * len(train_sizes) * len(seeds) * len(inc_grid)
    ordered: list[QuantumKernelResult | None] = []
    pending: list[tuple[int, QsvmSweepTask]] = []
    idx = 0

    for inc in inc_grid:
        for encoding in encodings:
            for seed in seeds:
                seed_subsets = training_subsets(
                    X_train_raw,
                    y_train,
                    sizes=int_sizes,
                    random_state=int(seed),
                )
                for tsz in train_sizes:
                    subset = seed_subsets["full"] if tsz == "full" else seed_subsets[int(tsz)]
                    ckey = (str(encoding), int(subset.size), int(seed), inc)
                    if ckey in cache:
                        ordered.append(cache[ckey])
                        idx += 1
                        continue

                    task = QsvmSweepTask(
                        encoding=str(encoding),
                        seed=int(seed),
                        X_train_raw=np.asarray(subset.X, dtype=np.int32),
                        y_train=np.asarray(subset.y),
                        train_size=int(subset.size),
                        include_nim_sum=bool(inc),
                        c_svc=float(c_svc),
                        estimator_mode=estimator_mode,
                        kernel_backend=kernel_backend,
                        shots=int(shots),
                    )
                    ordered.append(None)
                    pending.append((idx, task))
                    idx += 1

    if pending:

        def _run_serial(t: QsvmSweepTask) -> QuantumKernelResult:
            return execute_qsvm_sweep_task(
                t,
                X_test_raw,
                y_test,
                class_weight=class_weight,
                M=M,
                bits_per_heap=bits_per_heap,
                iqp_reps=iqp_reps,
                include_nim_sum=t.include_nim_sum,
                symmetry=symmetry,
                compute_win_rate=compute_win_rate,
                n_games_win_rate=n_games_win_rate,
                c_svc=float(c_svc),
                estimator_mode=estimator_mode,
                kernel_backend=kernel_backend,
                shots=int(shots),
            )

        def _after_task(_j: int, res: QuantumKernelResult) -> None:
            if mlflow_mod is not None:
                _log_mlflow_qsvm(
                    res,
                    mlflow_mod,
                    bits_per_heap=bits_per_heap,
                    iqp_reps=iqp_reps,
                    c_svc=float(c_svc),
                    estimator_mode=estimator_mode,
                    kernel_backend=kernel_backend,
                    shots=int(shots) if estimator_mode == "shot_binomial" else None,
                    run_name_prefix=mlflow_run_prefix,
                    extra_params=_mlflow_xp,
                )

        run_pending_grid_tasks(
            pending,
            ordered,
            run_serial=_run_serial,
            parallel_worker=_qsvm_sweep_worker,
            max_workers=max_workers,
            tqdm_desc=effective_tqdm_desc,
            initializer=_qsvm_sweep_pool_init,
            initargs=(
                X_test_raw,
                y_test,
                class_weight,
                M,
                bits_per_heap,
                iqp_reps,
                symmetry,
                compute_win_rate,
                n_games_win_rate,
                float(c_svc),
                estimator_mode,
                kernel_backend,
                int(shots),
            ),
            on_task_complete=_after_task,
        )

    sweep = QuantumKernelSweepResults()
    sweep.results = [r for r in ordered if r is not None]
    sweep.sweep_metadata = {
        "elapsed_wall_time_s": float(time.perf_counter() - sweep_started),
        "max_workers": (
            int(max_workers)
            if (max_workers is not None and max_workers > 1)
            else 1
        ),
        "n_tasks": int(total),
        "n_cached": int(total - len(pending)),
    }

    if verbose:
        _LOG.info("  QSVM sweep complete: %d/%d runs.", len(sweep.results), total)
    return sweep


__all__ = [
    "QsvmSweepTask",
    "execute_qsvm_sweep_task",
    "run_quantum_kernel_sweep",
]
