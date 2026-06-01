"""QSVM model, fitter, evaluator, and Nim-policy wrapper."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Sequence, cast

import numpy as np
import pandas as pd
from numpy.random import Generator
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

from qml_project.nim.data import normalise_states
from qml_project.nim.encoding import EncodingName, SymmetryMode
from qml_project.nim.game import (
    NimMove,
    NimState,
    apply_move,
    legal_moves,
    play_many,
    random_policy,
)
from qml_project.training.stats import (
    _grouped_bootstrap_summary,
    fit_power_law_learning_curve,
    sample_efficiency_stat_tests,
)

from .kernel import KernelBackend, KernelEstimatorMode, quantum_kernel_matrix


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
    c_svc: float = 1.0
    estimator_mode: KernelEstimatorMode = "exact_statevector"
    kernel_backend: KernelBackend = "manual"
    shots: int | None = None
    include_nim_sum: bool = True
    kernel_matrix_time_s: float = 0.0


@dataclass
class QuantumKernelSweepResults:
    """Aggregated QSVM sweep results."""

    results: list[QuantumKernelResult] = field(default_factory=list)
    sweep_metadata: dict[str, float | int | None] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        schema: tuple[str, ...] = (
            "pipeline",
            "encoding",
            "train_size",
            "seed",
            "symmetry",
            "include_nim_sum",
            "accuracy",
            "balanced_accuracy",
            "mcc",
            "f1",
            "precision",
            "recall",
            "train_time_s",
            "kernel_matrix_time_s",
            "inference_time_s",
            "win_rate",
            "c_svc",
            "estimator_mode",
            "kernel_backend",
            "shots",
        )
        rows: list[dict[str, float | int | str | None]] = []
        for r in self.results:
            row = asdict(r)
            row["pipeline"] = "qsvm"
            row.pop("cm", None)
            rows.append({key: cast(Any, row.get(key)) for key in schema})
        return pd.DataFrame(rows, columns=cast(Any, list(schema)))

    def summary(
        self,
        group_cols: Sequence[str] = ("encoding", "train_size", "symmetry", "include_nim_sum"),
        *,
        bootstrap_random_state: int = 42,
    ) -> pd.DataFrame:
        """Aggregate over seeds with mean/std and bootstrap confidence intervals."""
        return _grouped_bootstrap_summary(
            self.to_dataframe(),
            group_cols,
            (
                "accuracy",
                "balanced_accuracy",
                "mcc",
                "f1",
                "train_time_s",
                "kernel_matrix_time_s",
                "inference_time_s",
                "win_rate",
            ),
            bootstrap_random_state=bootstrap_random_state,
        )

    def statistical_tests(
        self,
        *,
        metrics: Sequence[str] = ("balanced_accuracy", "win_rate"),
        train_sizes: Sequence[int] | None = None,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """Paired Wilcoxon/effect-size tests across train sizes per encoding."""
        df = self.to_dataframe()
        if df.empty:
            return df
        if train_sizes is None:
            train_sizes = sorted(df["train_size"].dropna().unique().tolist())

        frames: list[pd.DataFrame] = []
        group_cols = ["encoding", "symmetry", "include_nim_sum"]
        for keys, g in df.groupby(group_cols, dropna=False):
            keys_tuple = keys if isinstance(keys, tuple) else (keys,)
            for metric in metrics:
                if metric not in g.columns:
                    continue
                stats = sample_efficiency_stat_tests(
                    g,
                    metric=metric,
                    train_sizes=train_sizes,
                    alpha=alpha,
                )
                if stats.empty:
                    continue
                stats = stats.copy()
                for col, value in zip(group_cols, keys_tuple):
                    stats[col] = value
                frames.append(stats)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def power_law_fits(
        self,
        *,
        metrics: Sequence[str] = ("balanced_accuracy", "win_rate"),
        train_sizes: Sequence[int] | None = None,
    ) -> pd.DataFrame:
        """Power-law fits of metric means over train sizes per encoding."""
        df = self.to_dataframe()
        if df.empty:
            return df
        if train_sizes is None:
            train_sizes = sorted(df["train_size"].dropna().unique().tolist())

        rows: list[dict[str, float | str]] = []
        for keys, g in df.groupby(["encoding", "symmetry", "include_nim_sum"], dropna=False):
            keys_tuple = keys if isinstance(keys, tuple) else (keys, "", True)
            encoding, symmetry, inc_ns = keys_tuple
            for metric in metrics:
                if metric not in g.columns:
                    continue
                means: list[float] = []
                valid_sizes: list[float] = []
                for size in train_sizes:
                    vals = g.loc[g["train_size"] == size, metric].dropna().to_numpy()
                    if vals.size == 0:
                        continue
                    valid_sizes.append(float(size))
                    means.append(float(np.mean(vals)))
                if len(valid_sizes) < 3:
                    continue
                fit = fit_power_law_learning_curve(valid_sizes, means)
                rows.append(
                    {
                        "encoding": str(encoding),
                        "symmetry": str(symmetry),
                        "include_nim_sum": bool(inc_ns),
                        "metric": metric,
                        **fit,
                        "n_points": float(len(valid_sizes)),
                    }
                )
        return pd.DataFrame(rows)


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
    c_svc: float = 1.0
    estimator_mode: KernelEstimatorMode = "exact_statevector"
    kernel_backend: KernelBackend = "manual"
    shots: int | None = None
    seed: int = 42

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
            estimator_mode=self.estimator_mode,
            kernel_backend=self.kernel_backend,
            shots=self.shots if self.shots is not None else 1024,
            seed=self.seed,
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
    c_svc: float = 1.0,
    estimator_mode: KernelEstimatorMode = "exact_statevector",
    kernel_backend: KernelBackend = "manual",
    shots: int = 1024,
) -> tuple[QuantumKernelSVMModel, float, float]:
    """Fit QSVM with a precomputed quantum kernel matrix.

    Returns
    -------
    model, train_wall_time_s, kernel_matrix_time_s
        Wall time includes kernel construction and ``SVC.fit``; kernel matrix time
        is only the Gram-matrix build (``quantum_kernel_matrix`` on train×train).
    """
    t0 = time.perf_counter()
    t_k0 = time.perf_counter()
    K_train = quantum_kernel_matrix(
        X_train_raw,
        X_train_raw,
        encoding=encoding,
        M=M,
        bits_per_heap=bits_per_heap,
        iqp_reps=iqp_reps,
        include_nim_sum=include_nim_sum,
        symmetry=symmetry,
        estimator_mode=estimator_mode,
        kernel_backend=kernel_backend,
        shots=shots,
        seed=random_state,
        validate=True,
    )
    kernel_matrix_time_s = float(time.perf_counter() - t_k0)
    model = SVC(
        kernel="precomputed",
        class_weight=class_weight,
        random_state=random_state,
        C=float(c_svc),
    )
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
            c_svc=float(c_svc),
            estimator_mode=estimator_mode,
            kernel_backend=kernel_backend,
            shots=int(shots) if estimator_mode == "shot_binomial" else None,
            seed=int(random_state),
        ),
        float(train_time),
        kernel_matrix_time_s,
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
    zd: int | str = 0
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
        c_svc=qsvm.c_svc,
        estimator_mode=qsvm.estimator_mode,
        kernel_backend=qsvm.kernel_backend,
        shots=qsvm.shots,
        include_nim_sum=qsvm.include_nim_sum,
        kernel_matrix_time_s=0.0,
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


def angle_features_for_vqc(states: np.ndarray, *, M_max: int = 7) -> np.ndarray:
    """Convenience helper for VQC comparison on the same train/test split."""
    return normalise_states(states, M_max=M_max) * np.pi


__all__ = [
    "QuantumKernelResult",
    "QuantumKernelSVMModel",
    "QuantumKernelSweepResults",
    "angle_features_for_vqc",
    "evaluate_qsvm_win_rate",
    "evaluate_quantum_kernel_svm",
    "fit_quantum_kernel_svm",
    "qsvm_policy",
]
