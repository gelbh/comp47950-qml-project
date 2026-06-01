"""Pickle-safe task dataclass and worker entry points for the classical sweep.

Top-level definitions are required so the multiprocessing pool can pickle
them under the ``spawn`` start method used on macOS and Windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, cast

import numpy as np

from qml_project.baselines.evaluation import ClassicalResult, evaluate_model
from qml_project.baselines.features import FeatureSet, prepare_features
from qml_project.baselines.models import create_models
from qml_project.baselines.nim_policy import evaluate_win_rate
from qml_project.nim.data import canonical_order


_classical_pool: dict[str, Any] = {}


def _make_feature_fn(
    feature_set: FeatureSet, M: int, *, symmetry: str = "none"
) -> Callable[[np.ndarray], np.ndarray]:
    """Create a feature-transform closure for use in game-play evaluation."""

    def fn(states: np.ndarray) -> np.ndarray:
        if symmetry == "canonical":
            states, _ = canonical_order(np.asarray(states, dtype=np.int32))
        return prepare_features(states, feature_set, M=M)

    return fn


@dataclass(frozen=True)
class ClassicalSweepTask:
    """One classical baseline run (raw train arrays + metadata)."""

    X_sub_raw: np.ndarray
    y_sub: np.ndarray
    model_name: str
    feature_set: str
    symmetry: str
    train_size: int
    seed: int
    compute_win_rate: bool
    n_games_win_rate: int
    c_svc: float = 1.0


def classical_sweep_pool_init(
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    M: int,
) -> None:
    """Initialise the subprocess pool state for parallel sweeps."""
    _classical_pool.clear()
    _classical_pool["X_test_raw"] = X_test_raw
    _classical_pool["y_test"] = y_test
    _classical_pool["M"] = int(M)


def execute_classical_sweep_task(
    task: ClassicalSweepTask,
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    M: int,
) -> ClassicalResult:
    """Train/evaluate one classical configuration (serial and parallel share this)."""
    fs = cast(FeatureSet, task.feature_set)
    if task.symmetry == "canonical":
        X_test_use, _ = canonical_order(np.asarray(X_test_raw, dtype=np.int32))
        X_sub_use, _ = canonical_order(np.asarray(task.X_sub_raw, dtype=np.int32))
    else:
        X_test_use = X_test_raw
        X_sub_use = task.X_sub_raw

    X_test_feat = prepare_features(X_test_use, fs, M=M)
    X_sub_feat = prepare_features(X_sub_use, fs, M=M)
    models = create_models(random_state=task.seed, M=M, c_svc=task.c_svc)
    model = models[task.model_name]
    result = evaluate_model(
        model,
        X_sub_feat,
        task.y_sub,
        X_test_feat,
        y_test,
        model_name=task.model_name,
    )
    result.seed = task.seed
    result.train_size = task.train_size
    result.feature_set = task.feature_set
    result.symmetry = task.symmetry
    result.regime = "ood"
    result.c_svc = float(task.c_svc)
    if task.compute_win_rate:
        feat_fn = _make_feature_fn(fs, M, symmetry=task.symmetry)
        result.win_rate = evaluate_win_rate(
            model,
            feat_fn,
            n_games=task.n_games_win_rate,
            k=3,
            M=M,
            seed=task.seed,
        )
    return result


def classical_sweep_worker(task: ClassicalSweepTask) -> ClassicalResult:
    """Subprocess entry point; consumes the pool-local test arrays."""
    return execute_classical_sweep_task(
        task,
        _classical_pool["X_test_raw"],
        _classical_pool["y_test"],
        _classical_pool["M"],
    )
