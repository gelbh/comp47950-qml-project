r"""Quantum-kernel SVM pipeline for Nim sample-efficiency experiments.

Implements the first quantum milestone from the project plan:

- Build quantum feature-map states for Nim encodings (angle, amplitude, binary,
  IQP parity).
- Compute kernel matrices
  ``k(x, x') = |<0|U^\dagger(x) U(x')|0>|^2 = |<psi(x)|psi(x')>|^2``.
- Train sklearn SVC with precomputed kernels.
- Run OOD sample-efficiency sweeps across training sizes and seeds.
"""

from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence, cast

import numpy as np
import pandas as pd
from numpy.random import Generator
from qiskit.quantum_info import Statevector
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.svm import SVC

from qml_project.nim.data import normalise_states, training_subsets
from qml_project.parallel_sweep import map_parallel_or_serial
from qml_project.nim.encoding import EncodingName, SymmetryMode, build_encoding_circuit
from qml_project.nim.game import NimMove, NimState, apply_move, legal_moves, play_many, random_policy


def _state_key(state: np.ndarray | Sequence[int]) -> tuple[int, ...]:
    """Return a stable hashable key for one Nim state."""
    arr = np.asarray(state, dtype=np.int32).ravel()
    return tuple(int(v) for v in arr)


def _build_statevector(
    state: np.ndarray | Sequence[int],
    *,
    encoding: EncodingName,
    M: int = 7,
    bits_per_heap: int = 3,
    iqp_reps: int = 2,
    include_nim_sum: bool = True,
    symmetry: SymmetryMode = "none",
) -> np.ndarray:
    """Build a feature-map statevector for one Nim state."""
    circuit = build_encoding_circuit(
        encoding,
        _state_key(state),
        M=M,
        bits_per_heap=bits_per_heap,
        iqp_reps=iqp_reps,
        include_nim_sum=include_nim_sum,
        symmetry=symmetry,
    )
    return Statevector.from_instruction(circuit).data


def quantum_kernel_matrix(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    encoding: EncodingName,
    M: int = 7,
    bits_per_heap: int = 3,
    iqp_reps: int = 2,
    include_nim_sum: bool = True,
    symmetry: SymmetryMode = "none",
) -> np.ndarray:
    """Compute the quantum kernel matrix between two state sets.

    Notes
    -----
    Uses exact statevector overlap:
    ``k(x, x') = |<psi(x)|psi(x')>|^2``.
    """
    X = np.asarray(X, dtype=np.int32)
    Y = np.asarray(Y, dtype=np.int32)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D arrays of raw heap states.")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same number of heaps/features.")

    cache: dict[tuple[int, ...], np.ndarray] = {}

    def get_sv(state: np.ndarray) -> np.ndarray:
        key = _state_key(state)
        if key not in cache:
            cache[key] = _build_statevector(
                key,
                encoding=encoding,
                M=M,
                bits_per_heap=bits_per_heap,
                iqp_reps=iqp_reps,
                include_nim_sum=include_nim_sum,
                symmetry=symmetry,
            )
        return cache[key]

    sv_x = [get_sv(X[i]) for i in range(X.shape[0])]
    sv_y = [get_sv(Y[j]) for j in range(Y.shape[0])]

    K = np.empty((X.shape[0], Y.shape[0]), dtype=np.float64)
    for i, a in enumerate(sv_x):
        overlaps = np.array([np.vdot(a, b) for b in sv_y], dtype=np.complex128)
        K[i] = np.abs(overlaps) ** 2
    return K


@dataclass
class QuantumKernelResult:
    """Result for one QSVM run (single encoding / train size / seed)."""

    encoding: EncodingName
    train_size: int
    seed: int
    accuracy: float
    balanced_accuracy: float
    mcc: float
    f1: float
    precision: float
    recall: float
    train_time_s: float
    inference_time_s: float
    symmetry: SymmetryMode = "none"
    win_rate: float | None = None
    cm: np.ndarray | None = None


@dataclass
class QuantumKernelSweepResults:
    """Aggregated QSVM sweep results."""

    results: list[QuantumKernelResult] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        rows: list[dict[str, float | int | str | None]] = []
        for r in self.results:
            rows.append(
                {
                    "pipeline": "qsvm",
                    "encoding": r.encoding,
                    "train_size": r.train_size,
                    "seed": r.seed,
                    "symmetry": r.symmetry,
                    "accuracy": r.accuracy,
                    "balanced_accuracy": r.balanced_accuracy,
                    "mcc": r.mcc,
                    "f1": r.f1,
                    "precision": r.precision,
                    "recall": r.recall,
                    "train_time_s": r.train_time_s,
                    "inference_time_s": r.inference_time_s,
                    "win_rate": r.win_rate,
                }
            )
        return pd.DataFrame(rows)

    def summary(
        self,
        group_cols: Sequence[str] = ("encoding", "train_size", "symmetry"),
    ) -> pd.DataFrame:
        df = self.to_dataframe()
        if df.empty:
            return df
        metric_cols = [
            "accuracy",
            "balanced_accuracy",
            "mcc",
            "f1",
            "train_time_s",
            "inference_time_s",
            "win_rate",
        ]
        grouped = df.groupby(list(group_cols), dropna=False)[metric_cols].agg(["mean", "std"])
        grouped.columns = [f"{m}_{s}" for m, s in grouped.columns]
        return grouped.reset_index()


@dataclass
class QuantumKernelSVMModel:
    """Trained QSVM wrapper that supports prediction on raw Nim states."""

    model: SVC
    X_train_raw: np.ndarray
    encoding: EncodingName
    M: int = 7
    bits_per_heap: int = 3
    iqp_reps: int = 2
    include_nim_sum: bool = True
    symmetry: SymmetryMode = "none"

    def kernel_to_train(self, X_new_raw: np.ndarray) -> np.ndarray:
        """Compute precomputed-kernel rows from new states to train states."""
        return quantum_kernel_matrix(
            X_new_raw,
            self.X_train_raw,
            encoding=self.encoding,
            M=self.M,
            bits_per_heap=self.bits_per_heap,
            iqp_reps=self.iqp_reps,
            include_nim_sum=self.include_nim_sum,
            symmetry=self.symmetry,
        )

    def predict_states(self, X_new_raw: np.ndarray) -> np.ndarray:
        """Predict labels for raw Nim states."""
        K_new = self.kernel_to_train(X_new_raw)
        return self.model.predict(K_new)


def fit_quantum_kernel_svm(
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    *,
    encoding: EncodingName,
    class_weight: str | dict[int, float] | None = "balanced",
    M: int = 7,
    bits_per_heap: int = 3,
    iqp_reps: int = 2,
    include_nim_sum: bool = True,
    symmetry: SymmetryMode = "none",
    random_state: int = 42,
) -> tuple[QuantumKernelSVMModel, float]:
    """Fit QSVM with a precomputed quantum kernel matrix."""
    t0 = time.perf_counter()
    K_train = quantum_kernel_matrix(
        X_train_raw,
        X_train_raw,
        encoding=encoding,
        M=M,
        bits_per_heap=bits_per_heap,
        iqp_reps=iqp_reps,
        include_nim_sum=include_nim_sum,
        symmetry=symmetry,
    )
    model = SVC(kernel="precomputed", class_weight=class_weight, random_state=random_state)
    model.fit(K_train, y_train)
    train_time = time.perf_counter() - t0
    return (
        QuantumKernelSVMModel(
            model=model,
            X_train_raw=np.asarray(X_train_raw, dtype=np.int32),
            encoding=encoding,
            M=M,
            bits_per_heap=bits_per_heap,
            iqp_reps=iqp_reps,
            include_nim_sum=include_nim_sum,
            symmetry=symmetry,
        ),
        float(train_time),
    )


def evaluate_quantum_kernel_svm(
    qsvm: QuantumKernelSVMModel,
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
) -> tuple[QuantumKernelResult, np.ndarray]:
    """Evaluate a trained QSVM model on raw Nim states."""
    t0 = time.perf_counter()
    y_pred = qsvm.predict_states(X_test_raw)
    inference_time = time.perf_counter() - t0
    zd = "warn"
    result = QuantumKernelResult(
        encoding=qsvm.encoding,
        train_size=len(qsvm.X_train_raw),
        seed=0,
        accuracy=float(accuracy_score(y_test, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_test, y_pred)),
        mcc=float(matthews_corrcoef(y_test, y_pred)),
        f1=float(f1_score(y_test, y_pred, average="binary", zero_division=zd)),
        precision=float(precision_score(y_test, y_pred, average="binary", zero_division=zd)),
        recall=float(recall_score(y_test, y_pred, average="binary", zero_division=zd)),
        train_time_s=0.0,
        inference_time_s=float(inference_time),
        symmetry=qsvm.symmetry,
        cm=confusion_matrix(y_test, y_pred),
    )
    return result, y_pred


def qsvm_policy(qsvm: QuantumKernelSVMModel) -> Callable[[NimState, Generator], NimMove]:
    """Wrap QSVM binary classifier as a Nim move policy."""

    def policy(state: NimState, rng: Generator) -> NimMove:
        moves = legal_moves(state)
        if len(moves) == 1:
            return moves[0]
        resulting_states = np.array([apply_move(state, m) for m in moves], dtype=np.int32)
        preds = qsvm.predict_states(resulting_states)
        good_mask = preds == 0
        if good_mask.any():
            good_idx = np.flatnonzero(good_mask)
            return moves[int(rng.choice(good_idx))]
        return moves[int(rng.integers(len(moves)))]

    return policy


def evaluate_qsvm_win_rate(
    qsvm: QuantumKernelSVMModel,
    *,
    n_games: int = 200,
    k: int = 3,
    M: int = 7,
    seed: int = 42,
) -> float:
    """Play QSVM policy vs random and return first-player win rate."""
    stats = play_many(qsvm_policy(qsvm), random_policy, n_games=n_games, k=k, M=M, seed=seed)
    return float(stats["win_rate_a"])  # type: ignore[arg-type]


def _load_qsvm_sweep_from_mlflow(
    experiment_name: str,
    encodings: Sequence[EncodingName],
    train_sizes: Sequence[int | str],
    seeds: Sequence[int],
    *,
    full_train_size: int,
    symmetry: SymmetryMode,
    include_nim_sum: bool,
    bits_per_heap: int,
    iqp_reps: int,
    compute_win_rate: bool,
) -> dict[tuple[str, int, int], QuantumKernelResult]:
    """Load QSVM sweep grid points from MLflow (newest run wins per key)."""
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

    wanted: set[tuple[str, int, int]] = set()
    for enc in encodings:
        for tsz in train_sizes:
            size = full_train_size if tsz == "full" else int(tsz)
            for seed in seeds:
                wanted.add((str(enc), size, int(seed)))

    inc_ns = "True" if include_nim_sum else "False"
    cache: dict[tuple[str, int, int], QuantumKernelResult] = {}

    for run in runs:
        if run.info.status != "FINISHED":
            continue
        p = run.data.params
        m = run.data.metrics
        if p.get("pipeline") != "qsvm":
            continue
        if (
            p.get("symmetry") != str(symmetry)
            or p.get("include_nim_sum") != inc_ns
            or str(p.get("bits_per_heap")) != str(bits_per_heap)
            or str(p.get("iqp_reps")) != str(iqp_reps)
        ):
            continue
        try:
            enc = p.get("encoding")
            train_size_int = int(p["train_size"])
            seed_int = int(p["seed"])
        except (KeyError, TypeError, ValueError):
            continue
        if not isinstance(enc, str):
            continue
        key = (enc, train_size_int, seed_int)
        if key not in wanted or key in cache:
            continue
        if compute_win_rate and "win_rate" not in m:
            continue
        win_rate: float | None = float(m["win_rate"]) if "win_rate" in m else None
        cache[key] = QuantumKernelResult(
            encoding=cast(EncodingName, enc),
            train_size=train_size_int,
            seed=seed_int,
            accuracy=float(m.get("accuracy", 0.0)),
            balanced_accuracy=float(m.get("balanced_accuracy", 0.0)),
            mcc=float(m.get("mcc", 0.0)),
            f1=float(m.get("f1", 0.0)),
            precision=float(m.get("precision", 0.0)),
            recall=float(m.get("recall", 0.0)),
            train_time_s=float(m.get("train_time_s", 0.0)),
            inference_time_s=float(m.get("inference_time_s", 0.0)),
            symmetry=symmetry,
            win_rate=win_rate,
            cm=None,
        )

    return cache


# ---------------------------------------------------------------------------
# Parallel sweep (top-level worker for multiprocessing spawn)
# ---------------------------------------------------------------------------

_qsvm_pool: dict[str, Any] = {}


@dataclass(frozen=True)
class QsvmSweepTask:
    """One QSVM grid point (materialised train subset)."""

    encoding: str
    seed: int
    X_train_raw: np.ndarray
    y_train: np.ndarray
    train_size: int


def _qsvm_sweep_pool_init(
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    class_weight: str | dict[int, float] | None,
    M: int,
    bits_per_heap: int,
    iqp_reps: int,
    include_nim_sum: bool,
    symmetry: SymmetryMode,
    compute_win_rate: bool,
    n_games_win_rate: int,
) -> None:
    _qsvm_pool.clear()
    _qsvm_pool["X_test_raw"] = X_test_raw
    _qsvm_pool["y_test"] = y_test
    _qsvm_pool["class_weight"] = class_weight
    _qsvm_pool["M"] = M
    _qsvm_pool["bits_per_heap"] = bits_per_heap
    _qsvm_pool["iqp_reps"] = iqp_reps
    _qsvm_pool["include_nim_sum"] = include_nim_sum
    _qsvm_pool["symmetry"] = symmetry
    _qsvm_pool["compute_win_rate"] = compute_win_rate
    _qsvm_pool["n_games_win_rate"] = n_games_win_rate


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
) -> QuantumKernelResult:
    """Fit and evaluate QSVM for one grid point."""
    encoding = cast(EncodingName, task.encoding)
    qsvm, train_time = fit_quantum_kernel_svm(
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
    )
    res, _ = evaluate_quantum_kernel_svm(qsvm, X_test_raw, y_test)
    res.seed = int(task.seed)
    res.train_size = int(task.train_size)
    res.train_time_s = float(train_time)
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
        include_nim_sum=_qsvm_pool["include_nim_sum"],
        symmetry=_qsvm_pool["symmetry"],
        compute_win_rate=_qsvm_pool["compute_win_rate"],
        n_games_win_rate=_qsvm_pool["n_games_win_rate"],
    )


def run_quantum_kernel_sweep(
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    *,
    encodings: Sequence[EncodingName] = ("angle", "amplitude", "binary", "iqp_parity"),
    train_sizes: Sequence[int | str] = (50, 100, "full"),
    seeds: Sequence[int] = tuple(range(10)),
    class_weight: str | dict[int, float] | None = "balanced",
    M: int = 7,
    bits_per_heap: int = 3,
    iqp_reps: int = 2,
    include_nim_sum: bool = True,
    symmetry: SymmetryMode = "none",
    compute_win_rate: bool = True,
    n_games_win_rate: int = 200,
    mlflow_experiment: str | None = None,
    use_cache: bool = True,
    force_rerun: bool = False,
    verbose: bool = True,
    max_workers: int | None = None,
    use_tqdm: bool = True,
) -> QuantumKernelSweepResults:
    """Run multi-seed QSVM sample-efficiency sweep (OOD-ready).

    When ``use_cache=True`` and ``mlflow_experiment`` is set, finished runs
    in that experiment matching the sweep grid are loaded from MLflow and
    only missing grid points are computed. Use ``force_rerun=True`` or
    ``use_cache=False`` after changing data or hyperparameters.

    When ``max_workers`` > 1, each grid point runs in a subprocess; MLflow
    logging happens in the parent. ``class_weight`` must be picklable (e.g.
    ``\"balanced\"`` or ``None``); dict weights are supported if hashable
    via pickling.
    """
    mlflow_mod = None
    if mlflow_experiment:
        try:
            import mlflow as _mlflow

            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
            )
            os.environ.setdefault("MLFLOW_TRACKING_URI", os.path.join(project_root, "mlruns"))
            _mlflow.set_experiment(mlflow_experiment)
            mlflow_mod = _mlflow
        except ImportError:
            warnings.warn("MLflow not installed; skipping logging.", stacklevel=2)

    full_train_size = int(len(X_train_raw))
    cache: dict[tuple[str, int, int], QuantumKernelResult] = {}
    if use_cache and not force_rerun and mlflow_mod is not None and mlflow_experiment:
        cache = _load_qsvm_sweep_from_mlflow(
            mlflow_experiment,
            encodings,
            train_sizes,
            seeds,
            full_train_size=full_train_size,
            symmetry=symmetry,
            include_nim_sum=include_nim_sum,
            bits_per_heap=bits_per_heap,
            iqp_reps=iqp_reps,
            compute_win_rate=compute_win_rate,
        )
        if verbose and cache:
            print(f"  Loaded {len(cache)} QSVM runs from MLflow cache.")

    int_sizes = [s for s in train_sizes if isinstance(s, int)]
    total = len(encodings) * len(train_sizes) * len(seeds)
    ordered: list[QuantumKernelResult | None] = []
    pending: list[tuple[int, QsvmSweepTask]] = []
    idx = 0

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
                ckey = (str(encoding), int(subset.size), int(seed))
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
                )
                ordered.append(None)
                pending.append((idx, task))
                idx += 1

    computed: list[QuantumKernelResult] = []
    if pending:
        tasks_only = [t for _, t in pending]
        if max_workers is None or max_workers <= 1:
            if use_tqdm:
                from tqdm.auto import tqdm as _tqdm

                computed = [
                    execute_qsvm_sweep_task(
                        t,
                        X_test_raw,
                        y_test,
                        class_weight=class_weight,
                        M=M,
                        bits_per_heap=bits_per_heap,
                        iqp_reps=iqp_reps,
                        include_nim_sum=include_nim_sum,
                        symmetry=symmetry,
                        compute_win_rate=compute_win_rate,
                        n_games_win_rate=n_games_win_rate,
                    )
                    for t in _tqdm(
                        tasks_only,
                        desc="QSVM sweep",
                        total=len(tasks_only),
                    )
                ]
            else:
                computed = [
                    execute_qsvm_sweep_task(
                        t,
                        X_test_raw,
                        y_test,
                        class_weight=class_weight,
                        M=M,
                        bits_per_heap=bits_per_heap,
                        iqp_reps=iqp_reps,
                        include_nim_sum=include_nim_sum,
                        symmetry=symmetry,
                        compute_win_rate=compute_win_rate,
                        n_games_win_rate=n_games_win_rate,
                    )
                    for t in tasks_only
                ]
        else:
            computed = map_parallel_or_serial(
                tasks_only,
                _qsvm_sweep_worker,
                max_workers=max_workers,
                use_tqdm=use_tqdm,
                tqdm_desc="QSVM sweep",
                initializer=_qsvm_sweep_pool_init,
                initargs=(
                    X_test_raw,
                    y_test,
                    class_weight,
                    M,
                    bits_per_heap,
                    iqp_reps,
                    include_nim_sum,
                    symmetry,
                    compute_win_rate,
                    n_games_win_rate,
                ),
            )

        for (i, _), res in zip(pending, computed, strict=True):
            ordered[i] = res

    sweep = QuantumKernelSweepResults()
    sweep.results = [r for r in ordered if r is not None]

    if mlflow_mod is not None and computed:
        for res in computed:
            _log_mlflow_qsvm(
                res,
                mlflow_mod,
                include_nim_sum=include_nim_sum,
                bits_per_heap=bits_per_heap,
                iqp_reps=iqp_reps,
            )

    if verbose:
        print(f"  QSVM sweep complete: {len(sweep.results)}/{total} runs.")
    return sweep


def _log_mlflow_qsvm(
    result: QuantumKernelResult,
    mlflow: Any,
    *,
    include_nim_sum: bool,
    bits_per_heap: int,
    iqp_reps: int,
) -> None:
    """Log one QSVM run to MLflow."""
    try:
        run_name = f"qsvm|{result.encoding}|n={result.train_size}|s={result.seed}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(
                {
                    "pipeline": "qsvm",
                    "encoding": result.encoding,
                    "train_size": result.train_size,
                    "seed": result.seed,
                    "symmetry": result.symmetry,
                    "include_nim_sum": include_nim_sum,
                    "bits_per_heap": bits_per_heap,
                    "iqp_reps": iqp_reps,
                }
            )
            metrics: dict[str, float] = {
                "accuracy": result.accuracy,
                "balanced_accuracy": result.balanced_accuracy,
                "mcc": result.mcc,
                "f1": result.f1,
                "precision": result.precision,
                "recall": result.recall,
                "train_time_s": result.train_time_s,
                "inference_time_s": result.inference_time_s,
            }
            if result.win_rate is not None:
                metrics["win_rate"] = result.win_rate
            mlflow.log_metrics(metrics)
    except Exception as exc:
        warnings.warn(f"MLflow logging failed: {exc}", stacklevel=2)


def build_kernel_pipeline_comparison(
    qsvm_results: QuantumKernelSweepResults,
    *,
    classical_df: pd.DataFrame | None = None,
    vqc_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create a unified comparison table: classical vs QSVM vs VQC.

    Parameters
    ----------
    qsvm_results
        Results from :func:`run_quantum_kernel_sweep`.
    classical_df
        Optional DataFrame from ``SweepResults.to_dataframe()``.
        Should include at least ``model``, ``train_size``, ``seed``,
        ``balanced_accuracy``, and ``win_rate``.
    vqc_df
        Optional DataFrame from ``SimulatedVQCSweepResults.to_dataframe()``.
        Should include ``train_size``, ``seed``, ``balanced_accuracy``, and
        ``win_rate``.
    """
    frames: list[pd.DataFrame] = []

    qdf = qsvm_results.to_dataframe().copy()
    if not qdf.empty:
        qdf["pipeline"] = "QSVM (Quantum Kernel)"
        qdf["model"] = qdf["encoding"].astype(str)
        qdf_subset = qdf.loc[
            :, ["pipeline", "model", "train_size", "seed", "balanced_accuracy", "win_rate"]
        ]
        frames.append(pd.DataFrame(qdf_subset))

    if classical_df is not None and not classical_df.empty:
        cdf = classical_df.copy()
        cdf = cdf[cdf["model"].isin(["SVM (RBF)", "SVM (Angle Kernel)"])].copy()
        cdf["pipeline"] = cdf["model"]
        cdf["model"] = cdf["model"].astype(str)
        cdf_subset = cdf.loc[
            :, ["pipeline", "model", "train_size", "seed", "balanced_accuracy", "win_rate"]
        ]
        frames.append(pd.DataFrame(cdf_subset))

    if vqc_df is not None and not vqc_df.empty:
        sdf = vqc_df.copy()
        sdf["pipeline"] = "VQC (Simulated)"
        if "ansatz" in sdf.columns:
            sdf["model"] = sdf["ansatz"].astype(str)
        else:
            sdf["model"] = "vqc"
        sdf_subset = sdf.loc[
            :, ["pipeline", "model", "train_size", "seed", "balanced_accuracy", "win_rate"]
        ]
        frames.append(pd.DataFrame(sdf_subset))

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)
    grouped = (
        merged.groupby(["pipeline", "model", "train_size"], dropna=False)[
            ["balanced_accuracy", "win_rate"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped.columns = [
        "pipeline",
        "model",
        "train_size",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "win_rate_mean",
        "win_rate_std",
    ]
    return grouped.sort_values(["train_size", "pipeline", "model"]).reset_index(drop=True)


def angle_features_for_vqc(states: np.ndarray, *, M_max: int = 7) -> np.ndarray:
    """Convenience helper for VQC comparison on the same train/test split."""
    return normalise_states(states, M_max=M_max) * np.pi
