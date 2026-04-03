"""Classical ML baselines for the Nim QML project.

Provides the full classical pipeline: four classifiers (SVM-RBF,
Random Forest, Logistic Regression, SVM with angle-encoding kernel),
five feature sets (raw, heap_parity, pairwise_xor, bit_parity, full parity),
S_3 symmetry augmentation, multi-seed sweeps, win-rate evaluation via game
play, and MLflow logging.

OOD design: train on M≤5 (subsets 50, 100, full=215), test on M>5
(296 states). Pass OOD train/test arrays to :func:`run_classical_sweep`.

Primary target: state → win/loss classification.
"""

from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence

import numpy as np
import pandas as pd
from numpy.random import Generator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel
from sklearn.svm import SVC

from qml_project.nim.data import augment_s3, normalise_states
from qml_project.nim.game import (
    NimMove,
    NimState,
    apply_move,
    is_terminal,
    legal_moves,
    nim_sum,
)

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

FeatureSet = Literal["raw", "parity", "heap_parity", "pairwise_xor", "bit_parity"]

ABLATION_FEATURE_SETS: tuple[FeatureSet, ...] = (
    "raw", "heap_parity", "pairwise_xor", "bit_parity", "parity",
)

# Parity-style feature sets only (excludes "raw" to avoid overlap with main sweep)
ABLATION_FEATURE_SETS_NO_RAW: tuple[FeatureSet, ...] = tuple(
    fs for fs in ABLATION_FEATURE_SETS if fs != "raw"
)

FEATURE_SET_DESCRIPTIONS: dict[str, str] = {
    "raw": "Normalised heaps (3)",
    "heap_parity": "+ heap parities (6)",
    "pairwise_xor": "+ pairwise XOR (6)",
    "bit_parity": "+ column bit parities (6)",
    "parity": "All parity features (12)",
}


def _heap_parities(states: np.ndarray) -> np.ndarray:
    """Per-heap parities: ``h_i mod 2``."""
    return (states % 2).astype(np.float64)


def _pairwise_xor(states: np.ndarray, M: int) -> np.ndarray:
    """Pairwise XOR of heap sizes, normalised by *M*."""
    n, k = states.shape
    pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            pairs.append((states[:, i] ^ states[:, j]).astype(np.float64) / M)
    return np.column_stack(pairs) if pairs else np.empty((n, 0))


def _bit_parities(states: np.ndarray, M: int) -> np.ndarray:
    """Column-wise bit parities (individual bits of the Nim-sum)."""
    n, k = states.shape
    n_bits = int(np.ceil(np.log2(M + 1)))
    bp = np.zeros((n, n_bits), dtype=np.float64)
    for b in range(n_bits):
        col_xor = np.zeros(n, dtype=np.int32)
        for i in range(k):
            col_xor ^= (states[:, i] >> b) & 1
        bp[:, b] = col_xor.astype(np.float64)
    return bp


def engineer_parity_features(states: np.ndarray, *, M: int = 7) -> np.ndarray:
    """Add parity / XOR features to raw heap-size arrays.

    Appends to each row:
      - Heap parities: ``h_i mod 2`` for each heap  (k features)
      - Pairwise XOR:  ``h_i ⊕ h_j`` for all pairs  (k*(k-1)/2 features)
      - Column-wise bit parities: XOR of bit *b* across all heaps
        for each bit position (``ceil(log2(M+1))`` features)

    Parameters
    ----------
    states : np.ndarray, shape ``(n, k)``
        Raw (unnormalised) integer heap sizes.
    M : int
        Maximum heap size (determines number of bit columns).

    Returns
    -------
    np.ndarray, shape ``(n, k + k + k*(k-1)/2 + n_bits)``
        Normalised heaps concatenated with engineered features.
    """
    norm = states.astype(np.float64) / M
    return np.hstack([
        norm, _heap_parities(states), _pairwise_xor(states, M), _bit_parities(states, M),
    ])


def prepare_features(
    states: np.ndarray,
    feature_set: FeatureSet = "raw",
    *,
    M: int = 7,
) -> np.ndarray:
    """Transform raw heap sizes into features for a given feature set.

    Parameters
    ----------
    states : np.ndarray, shape ``(n, k)``
        Raw integer heap sizes.
    feature_set : FeatureSet
        ``"raw"``          — normalised heap sizes only (3 features).
        ``"heap_parity"``  — raw + per-heap parities (6 features).
        ``"pairwise_xor"`` — raw + pairwise XOR (6 features).
        ``"bit_parity"``   — raw + column-wise bit parities (6 features).
        ``"parity"``       — all of the above (12 features).
    M : int
        Maximum heap size for normalisation.
    """
    if feature_set == "raw":
        return normalise_states(states, M_max=M)
    if feature_set == "parity":
        return engineer_parity_features(states, M=M)
    norm = states.astype(np.float64) / M
    if feature_set == "heap_parity":
        return np.hstack([norm, _heap_parities(states)])
    if feature_set == "pairwise_xor":
        return np.hstack([norm, _pairwise_xor(states, M)])
    if feature_set == "bit_parity":
        return np.hstack([norm, _bit_parities(states, M)])
    raise ValueError(f"Unknown feature_set: {feature_set!r}")


# ---------------------------------------------------------------------------
# Quantum-inspired kernels
# ---------------------------------------------------------------------------


def angle_encoding_kernel(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    M: int = 7,
) -> np.ndarray:
    r"""Kernel mimicking the angle-encoding quantum feature map (product state).

    Computes

    .. math::

        k(\mathbf x, \mathbf x')
        = \prod_{i=1}^{k} \cos^2\!\Bigl(\frac{(x_i - x'_i)\,\pi}{2}\Bigr)

    where :math:`x_i = h_i / M` are normalised heap sizes.  This equals
    :math:`|\langle\psi(\mathbf x)|\psi(\mathbf x')\rangle|^2` for the
    product-state encoding
    :math:`|\psi(\mathbf x)\rangle = \bigotimes_i R_Y(h_i\pi/M)|0\rangle`.

    Parameters
    ----------
    X, Y : np.ndarray
        Feature arrays of normalised heap sizes, shapes ``(n, d)`` and
        ``(m, d)``.
    M : int
        Present for API consistency; the features must already be normalised
        by *M* (i.e. values in [0, 1]).

    Returns
    -------
    np.ndarray, shape ``(n, m)``
        Kernel (Gram) matrix.
    """
    diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]  # (n, m, d)
    cos_sq = np.cos(diff * np.pi / 2) ** 2
    return np.prod(cos_sq, axis=2)


def _make_angle_kernel(M: int = 7):
    """Return a callable ``(X, Y) -> K`` suitable for ``SVC(kernel=...)``."""
    def _kernel(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return angle_encoding_kernel(X, Y, M=M)
    return _kernel


def centered_kernel_alignment(K1: np.ndarray, K2: np.ndarray) -> float:
    """Compute centered kernel alignment between two Gram matrices.

    The value is in ``[-1, 1]`` when both kernels are centered:

    .. math::

        \\mathrm{CKA}(K_1, K_2) =
        \\frac{\\langle K_{1,c}, K_{2,c}\\rangle_F}
             {\\|K_{1,c}\\|_F\\,\\|K_{2,c}\\|_F}

    where :math:`K_c = HKH` and :math:`H = I - \\frac{1}{n}\\mathbf 1\\mathbf 1^T`.
    """
    if K1.shape != K2.shape:
        raise ValueError("K1 and K2 must have the same shape.")
    n = K1.shape[0]
    if n != K1.shape[1]:
        raise ValueError("Kernel matrices must be square.")

    H = np.eye(n) - np.ones((n, n), dtype=np.float64) / n
    K1c = H @ K1 @ H
    K2c = H @ K2 @ H
    num = float(np.sum(K1c * K2c))
    den = float(np.linalg.norm(K1c, ord="fro") * np.linalg.norm(K2c, ord="fro"))
    if den == 0.0:
        return 0.0
    return num / den


def label_kernel_binary(y: np.ndarray) -> np.ndarray:
    """Return binary target kernel ``yy^T`` using labels in {0, 1}."""
    y = np.asarray(y).ravel()
    labels = set(np.unique(y).tolist())
    if not labels.issubset({0, 1}):
        raise ValueError("y must contain binary labels encoded as 0/1.")
    y_pm = 2.0 * y.astype(np.float64) - 1.0
    return np.outer(y_pm, y_pm)


def kernel_class_separation(
    K: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    """Summarise within-class vs between-class similarity in a kernel matrix."""
    if K.shape[0] != K.shape[1]:
        raise ValueError("K must be a square Gram matrix.")
    y = np.asarray(y).ravel()
    if len(y) != K.shape[0]:
        raise ValueError("y length must match K size.")

    same = y[:, None] == y[None, :]
    diff = ~same
    diag = np.eye(len(y), dtype=bool)
    same_wo_diag = same & (~diag)

    same_vals = K[same_wo_diag]
    diff_vals = K[diff]
    mean_same = float(np.mean(same_vals)) if same_vals.size else 0.0
    mean_diff = float(np.mean(diff_vals)) if diff_vals.size else 0.0
    return {
        "mean_same_class": mean_same,
        "mean_diff_class": mean_diff,
        "gap_same_minus_diff": mean_same - mean_diff,
    }


def compare_kernels_for_nim(
    X: np.ndarray,
    y: np.ndarray,
    *,
    M: int = 7,
    rbf_gamma: float = 1.0,
    poly_degree: int = 2,
    poly_gamma: float = 1.0,
    poly_coef0: float = 1.0,
) -> pd.DataFrame:
    """Compare angle, RBF, and polynomial kernels on Nim states.

    Returns a tidy DataFrame with quantitative diagnostics:
    centered alignment to the binary target kernel and same-vs-different class
    similarity gaps.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).ravel()
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if len(y) != len(X):
        raise ValueError("y length must match number of rows in X.")

    K_angle = angle_encoding_kernel(X, X, M=M)
    K_rbf = rbf_kernel(X, X, gamma=rbf_gamma)
    K_poly = polynomial_kernel(X, X, degree=poly_degree, gamma=poly_gamma, coef0=poly_coef0)
    K_target = label_kernel_binary(y)

    rows: list[dict[str, float | str]] = []
    kernels = {
        "angle": K_angle,
        "rbf": K_rbf,
        "poly": K_poly,
    }
    for name, K in kernels.items():
        sep = kernel_class_separation(K, y)
        rows.append({
            "kernel": name,
            "cka_to_target": centered_kernel_alignment(K, K_target),
            "mean_same_class": sep["mean_same_class"],
            "mean_diff_class": sep["mean_diff_class"],
            "gap_same_minus_diff": sep["gap_same_minus_diff"],
            "trace": float(np.trace(K)),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------


def create_models(
    random_state: int = 42,
    *,
    class_weight: str | dict | None = "balanced",
    M: int = 7,
) -> dict[str, Any]:
    """Create classifiers with class weighting.

    Returns a dict mapping model name to an unfitted sklearn estimator.
    Includes the three standard classifiers plus a quantum-inspired
    angle-encoding kernel SVM.
    """
    return {
        "SVM (RBF)": SVC(
            kernel="rbf",
            class_weight=class_weight,
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight=class_weight,
            random_state=random_state,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight=class_weight,
            random_state=random_state,
        ),
        "SVM (Angle Kernel)": SVC(
            kernel=_make_angle_kernel(M),
            class_weight=class_weight,
            random_state=random_state,
        ),
    }


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


@dataclass
class ClassicalResult:
    """Full evaluation result for a single classical model run.

    When loaded from MLflow cache, ``cm`` and ``y_pred`` are None
    (not logged to MLflow). Downstream code should check before using.
    """

    model_name: str
    accuracy: float
    balanced_accuracy: float
    mcc: float
    f1: float
    precision: float
    recall: float
    cm: np.ndarray | None = None
    y_pred: np.ndarray | None = None
    train_time_s: float = 0.0
    inference_time_s: float = 0.0
    # Sweep metadata
    seed: int = 42
    train_size: int | str = "full"
    feature_set: str = "raw"
    symmetry: str = "none"
    regime: str = "ood"
    win_rate: float | None = None


def evaluate_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    model_name: str = "",
) -> ClassicalResult:
    """Train a model and compute full evaluation metrics."""
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    inference_time = time.perf_counter() - t0

    zd = "warn"
    return ClassicalResult(
        model_name=model_name,
        accuracy=float(accuracy_score(y_test, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_test, y_pred)),
        mcc=float(matthews_corrcoef(y_test, y_pred)),
        f1=float(f1_score(y_test, y_pred, average="binary", zero_division=zd)),
        precision=float(precision_score(y_test, y_pred, average="binary", zero_division=zd)),
        recall=float(recall_score(y_test, y_pred, average="binary", zero_division=zd)),
        cm=confusion_matrix(y_test, y_pred),
        y_pred=y_pred,
        train_time_s=float(train_time),
        inference_time_s=float(inference_time),
    )


# ---------------------------------------------------------------------------
# Win-rate evaluation via game play
# ---------------------------------------------------------------------------


def model_policy(
    model: Any,
    feature_fn: Callable[[np.ndarray], np.ndarray],
    *,
    k: int = 3,
    M: int = 7,
) -> Callable[[NimState, Generator], NimMove]:
    """Wrap a trained win/loss classifier as a Nim policy.

    For each legal move, evaluates the resulting state with the model.
    Picks a move that leads to a state the model predicts as *losing*
    (for the opponent).  Falls back to a random legal move if none.
    """

    def policy(state: NimState, rng: Generator) -> NimMove:
        moves = legal_moves(state)
        if len(moves) == 1:
            return moves[0]

        resulting_states = np.array(
            [apply_move(state, m) for m in moves], dtype=np.int32,
        )
        X = feature_fn(resulting_states)
        preds = model.predict(X)

        # Moves where model predicts resulting state is losing (good for us)
        good_mask = preds == 0
        if good_mask.any():
            good_indices = np.flatnonzero(good_mask)
            return moves[int(rng.choice(good_indices))]

        # Fallback: random legal move
        return moves[int(rng.integers(len(moves)))]

    return policy


def evaluate_win_rate(
    model: Any,
    feature_fn: Callable[[np.ndarray], np.ndarray],
    *,
    n_games: int = 500,
    k: int = 3,
    M: int = 7,
    seed: int = 42,
) -> float:
    """Play the model (as first player) vs random and return win rate."""
    from qml_project.nim.game import play_many, random_policy

    pol = model_policy(model, feature_fn, k=k, M=M)
    stats = play_many(pol, random_policy, n_games=n_games, k=k, M=M, seed=seed)
    return float(stats["win_rate_a"])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Multi-configuration sweep
# ---------------------------------------------------------------------------


@dataclass
class SweepConfig:
    """A single configuration in the classical baseline sweep."""

    model_name: str
    feature_set: FeatureSet
    symmetry: str  # "none" or "augmented"
    train_size: int | str
    seed: int


@dataclass
class SweepResults:
    """Aggregated results from the full classical baseline sweep."""

    results: list[ClassicalResult] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a tidy DataFrame (one row per run)."""
        rows = []
        for r in self.results:
            rows.append({
                "model": r.model_name,
                "feature_set": r.feature_set,
                "symmetry": r.symmetry,
                "train_size": r.train_size,
                "seed": r.seed,
                "regime": r.regime,
                "accuracy": r.accuracy,
                "balanced_accuracy": r.balanced_accuracy,
                "mcc": r.mcc,
                "f1": r.f1,
                "precision": r.precision,
                "recall": r.recall,
                "train_time_s": r.train_time_s,
                "inference_time_s": r.inference_time_s,
                "win_rate": r.win_rate,
            })
        return pd.DataFrame(rows)

    def summary(
        self,
        group_cols: Sequence[str] = ("model", "feature_set", "symmetry", "train_size", "regime"),
    ) -> pd.DataFrame:
        """Aggregate over seeds: mean +/- std for each metric."""
        df = self.to_dataframe()
        metric_cols = [
            "accuracy", "balanced_accuracy", "mcc", "f1",
            "train_time_s", "win_rate",
        ]
        existing = [c for c in metric_cols if c in df.columns]
        grouped = df.groupby(list(group_cols))[existing].agg(["mean", "std"])
        grouped.columns = [f"{m}_{s}" for m, s in grouped.columns]
        return grouped.reset_index()


def _make_feature_fn(
    feature_set: FeatureSet, M: int,
) -> Callable[[np.ndarray], np.ndarray]:
    """Create a feature-transform closure for use in game-play evaluation."""
    def fn(states: np.ndarray) -> np.ndarray:
        return prepare_features(states, feature_set, M=M)
    return fn


def _load_sweep_from_mlflow(
    experiment_name: str,
    model_names: Sequence[str],
    feature_sets: Sequence[FeatureSet],
    symmetry_variants: Sequence[str],
    train_sizes: Sequence[int | str],
    seeds: Sequence[int],
    regime: str,
    *,
    full_train_size: int,
) -> dict[tuple[str, str, str, int, int], ClassicalResult]:
    """Load classical sweep results from MLflow runs (cache lookup).

    Returns a dict keyed by (model_name, feature_set, symmetry, train_size, seed).
    Only includes runs that match the requested grid and regime; when multiple
    runs exist for the same params, the latest (by end_time) is used.
    """
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

    # Set of keys we want (train_size as int; "full" -> full_train_size)
    wanted: set[tuple[str, str, str, int, int]] = set()
    for model in model_names:
        for fs in feature_sets:
            for sym in symmetry_variants:
                for tsz in train_sizes:
                    size = full_train_size if tsz == "full" else int(tsz)
                    for seed in seeds:
                        wanted.add((model, fs, sym, size, seed))

    cache: dict[tuple[str, str, str, int, int], ClassicalResult] = {}
    for run in runs:
        if run.info.status != "FINISHED":
            continue
        params = run.data.params
        metrics = run.data.metrics
        model = params.get("model")
        fs = params.get("feature_set")
        sym = params.get("symmetry")
        train_size_str = params.get("train_size")
        seed_str = params.get("seed")
        reg = params.get("regime")
        if (
            model is None
            or fs is None
            or sym is None
            or train_size_str is None
            or seed_str is None
            or reg is None
            or reg != regime
        ):
            continue
        try:
            train_size_int = int(train_size_str)
            seed_int = int(seed_str)
        except (TypeError, ValueError):
            continue
        # Narrow types after None checks (params from MLflow are string values)
        assert isinstance(model, str) and isinstance(fs, str) and isinstance(sym, str)
        key: tuple[str, str, str, int, int] = (model, fs, sym, train_size_int, seed_int)
        if key not in wanted or key in cache:
            continue
        cache[key] = ClassicalResult(
            model_name=model,
            accuracy=float(metrics.get("accuracy", 0.0)),
            balanced_accuracy=float(metrics.get("balanced_accuracy", 0.0)),
            mcc=float(metrics.get("mcc", 0.0)),
            f1=float(metrics.get("f1", 0.0)),
            precision=float(metrics.get("precision", 0.0)),
            recall=float(metrics.get("recall", 0.0)),
            cm=None,
            y_pred=None,
            train_time_s=float(metrics.get("train_time_s", 0.0)),
            inference_time_s=float(metrics.get("inference_time_s", 0.0)),
            seed=seed_int,
            train_size=train_size_int,
            feature_set=fs,
            symmetry=sym,
            regime=reg,
            win_rate=metrics.get("win_rate"),
        )
    return cache


def run_classical_sweep(
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    *,
    model_names: Sequence[str] = ("SVM (RBF)", "Random Forest", "Logistic Regression"),
    feature_sets: Sequence[FeatureSet] = ("raw", "parity"),
    symmetry_variants: Sequence[str] = ("none", "augmented"),
    train_sizes: Sequence[int | str] = (50, 100, "full"),
    seeds: Sequence[int] = tuple(range(10)),
    M: int = 7,
    compute_win_rate: bool = True,
    n_games_win_rate: int = 500,
    mlflow_experiment: str | None = None,
    use_cache: bool = True,
    force_rerun: bool = False,
    verbose: bool = True,
) -> SweepResults:
    """Run the full classical baseline sweep over configurations.

    When ``use_cache=True`` and ``mlflow_experiment`` is set, existing runs
    in that experiment that match the sweep grid are loaded from MLflow
    instead of re-running. Pass ``force_rerun=True`` or ``use_cache=False``
    to ignore cache (e.g. after changing train/test data).

    Parameters
    ----------
    X_train_raw, y_train : np.ndarray
        Full training arrays (raw integer heap sizes and labels).
    X_test_raw, y_test : np.ndarray
        Test arrays (raw integer heap sizes and labels).
    model_names : sequence of str
        Which models to include.
    feature_sets : sequence of FeatureSet
        Feature engineering variants.
    symmetry_variants : sequence of str
        ``"none"`` or ``"augmented"`` (S_3 data augmentation).
    train_sizes : sequence of int or str
        Training-size subsets.  ``"full"`` uses the entire training set.
    seeds : sequence of int
        Random seeds for model initialisation and sub-sampling.
        Each seed produces a different stratified subsample of the
        training data, so variance reflects both data sampling and
        (for non-deterministic models) model randomness.
    M : int
        Maximum heap size (for feature engineering and win-rate eval).
    compute_win_rate : bool
        Whether to evaluate win rate via game play (slower).
    n_games_win_rate : int
        Games per win-rate evaluation.
    mlflow_experiment : str or None
        If provided, log each run to this MLflow experiment.
    use_cache : bool
        If True and mlflow_experiment is set, load matching runs from MLflow
        and only run missing grid points. Default True.
    force_rerun : bool
        If True, ignore cache and run all grid points. Default False.
    verbose : bool
        Print progress.
    """
    from qml_project.nim.data import training_subsets

    # MLflow setup
    mlflow_mod = None
    if mlflow_experiment:
        try:
            import mlflow as _mlflow
            _project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
            )
            os.environ.setdefault(
                "MLFLOW_TRACKING_URI", os.path.join(_project_root, "mlruns")
            )
            _mlflow.set_experiment(mlflow_experiment)
            mlflow_mod = _mlflow
        except ImportError:
            warnings.warn("MLflow not installed; skipping logging.", stacklevel=2)

    # Load cache when use_cache and experiment set and not forcing rerun
    cache: dict[tuple[str, str, str, int, int], ClassicalResult] = {}
    if use_cache and not force_rerun and mlflow_mod and mlflow_experiment:
        full_train_size = len(X_train_raw)
        cache = _load_sweep_from_mlflow(
            mlflow_experiment,
            model_names,
            feature_sets,
            symmetry_variants,
            train_sizes,
            seeds,
            "ood",
            full_train_size=full_train_size,
        )
        if verbose and cache:
            print(f"  Loaded {len(cache)} runs from MLflow cache.")

    sweep = SweepResults()
    total = (
        len(model_names) * len(feature_sets) * len(symmetry_variants)
        * len(train_sizes) * len(seeds)
    )
    done = 0

    for model_name in model_names:
        for fs in feature_sets:
            X_test_feat = prepare_features(X_test_raw, fs, M=M)

            for sym in symmetry_variants:
                for seed in seeds:
                    # Per-seed subsets so variance reflects data sampling
                    int_sizes = [s for s in train_sizes if isinstance(s, int)]
                    seed_subsets = training_subsets(
                        X_train_raw, y_train, sizes=int_sizes,
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
                        cache_key = (model_name, fs, sym, result_train_size, seed)
                        if cache_key in cache:
                            sweep.results.append(cache[cache_key])
                            done += 1
                            if verbose and done % 20 == 0:
                                print(f"  [{done}/{total}] runs complete")
                            continue

                        # Apply symmetry augmentation
                        if sym == "augmented":
                            X_sub_use, y_sub_use = augment_s3(
                                X_sub, y_sub, deduplicate=True,
                            )
                        else:
                            X_sub_use, y_sub_use = X_sub, y_sub

                        X_sub_feat = prepare_features(X_sub_use, fs, M=M)

                        models = create_models(random_state=seed, M=M)
                        model = models[model_name]

                        result = evaluate_model(
                            model,
                            X_sub_feat, y_sub_use,
                            X_test_feat, y_test,
                            model_name=model_name,
                        )
                        result.seed = seed
                        result.train_size = (
                            tsz if isinstance(tsz, int) else len(X_sub)
                        )
                        result.feature_set = fs
                        result.symmetry = sym
                        result.regime = "ood"

                        # Win rate
                        if compute_win_rate:
                            feat_fn = _make_feature_fn(fs, M)
                            result.win_rate = evaluate_win_rate(
                                model, feat_fn, n_games=n_games_win_rate,
                                k=3, M=M, seed=seed,
                            )

                        sweep.results.append(result)

                        # MLflow
                        if mlflow_mod is not None:
                            _log_mlflow_run(result, mlflow_mod)

                        done += 1
                        if verbose and done % 20 == 0:
                            print(f"  [{done}/{total}] runs complete")

    if verbose:
        print(f"  Sweep complete: {done} runs.")
    return sweep


def _log_mlflow_run(result: ClassicalResult, mlflow: Any) -> None:
    """Log a single classical result to MLflow."""
    try:
        run_name = (
            f"{result.model_name}|{result.feature_set}|{result.symmetry}"
            f"|n={result.train_size}|s={result.seed}"
        )
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                "pipeline": "classical",
                "model": result.model_name,
                "feature_set": result.feature_set,
                "symmetry": result.symmetry,
                "train_size": result.train_size,
                "seed": result.seed,
                "regime": result.regime,
            })
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


# ---------------------------------------------------------------------------
# Convenience: single-model quick evaluation
# ---------------------------------------------------------------------------


def run_baseline(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Train a model and return metrics dict (legacy interface).

    Prefer :func:`evaluate_model` or :func:`run_classical_sweep` for new
    code.
    """
    result = evaluate_model(
        model, X_train, y_train, X_test, y_test,
    )
    return {
        "accuracy": result.accuracy,
        "balanced_accuracy": result.balanced_accuracy,
        "mcc": result.mcc,
        "f1_macro": result.f1,
        "precision_macro": result.precision,
        "recall_macro": result.recall,
        "confusion_matrix": result.cm,
        "y_pred": result.y_pred,
        "n_classes": len(np.unique(y_train)),
        "train_time_s": result.train_time_s,
        "inference_time_s": result.inference_time_s,
    }
