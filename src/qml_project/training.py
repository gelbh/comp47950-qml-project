"""
Simulation training for the variational quantum classifier.

Implements:
  - COBYLA-based optimisation (gradient-free).
  - Progressive shot schedule: 250 → 500 → 750 shots by evaluation number.
  - Multi-seed experiments for variance analysis.
  - Optional noise model support via Qiskit Aer.

Designed to work with Qiskit ≥ 2.0 primitives (V2 sampler interface).
"""

from __future__ import annotations

import itertools
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef

from qiskit.primitives import StatevectorSampler
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.primitives.containers.sampler_pub import SamplerPub

from qml_project.circuit import (
    VariationalClassifier,
    batch_loss,
    build_circuit,
    counts_to_class_probs,
    counts_to_z_expectation,
    predict_batch,
)
from qml_project.parallel_sweep import map_parallel_or_serial
from qml_project.nim.data import training_subsets
from qml_project.nim.game import (
    NimMove,
    NimState,
    Policy,
    apply_move,
    legal_moves,
    play_many,
    random_policy,
)

MeasurementObservable = Literal["bitstring_probs", "z_expectation"]
DecisionRule = Literal["argmax", "expectation_threshold"]
LossName = Literal[
    "softmax_nll",
    "cross_entropy_expectation",
    "hinge_expectation",
]

# ---------------------------------------------------------------------------
# Data classes for results
# ---------------------------------------------------------------------------


@dataclass
class TrainingHistory:
    """Records training progress across optimiser evaluations."""

    losses: list[float] = field(default_factory=list)
    train_accuracies: list[float] = field(default_factory=list)
    test_accuracies: list[float] = field(default_factory=list)
    eval_numbers: list[int] = field(default_factory=list)
    shot_counts: list[int] = field(default_factory=list)
    best_weights: np.ndarray | None = None
    best_loss: float = float("inf")
    total_training_time: float = 0.0
    total_evals: int = 0


@dataclass
class ExperimentResult:
    """Result from a single training run (one seed)."""

    seed: int
    best_weights: np.ndarray
    history: TrainingHistory
    test_accuracy: float
    test_predictions: np.ndarray
    test_class_probs: np.ndarray
    training_time: float
    inference_time: float
    balanced_accuracy: float | None = None
    mcc: float | None = None


@dataclass
class MultiSeedSummary:
    """Aggregated results across multiple random seeds."""

    per_seed: list[ExperimentResult]
    test_accuracy_mean: float
    test_accuracy_std: float
    test_accuracy_min: float
    test_accuracy_max: float
    training_time_mean: float
    inference_time_mean: float
    n_seeds: int


@dataclass
class SimulatedVQCRunResult:
    """Single simulated VQC run in the OOD sample-efficiency sweep."""

    train_size: int
    seed: int
    test_accuracy: float
    balanced_accuracy: float
    mcc: float
    win_rate: float | None
    training_time: float
    inference_time: float
    final_loss: float
    ansatz: str
    observable: MeasurementObservable
    decision_rule: DecisionRule
    loss_name: LossName


@dataclass(frozen=True)
class VqcOodSweepTask:
    """One simulated VQC OOD run (encoded train subset + metadata)."""

    subset_X: np.ndarray
    subset_y: np.ndarray
    seed: int
    train_size: int


@dataclass
class SimulatedVQCSweepResults:
    """Collection of simulated VQC runs across train sizes and seeds."""

    results: list[SimulatedVQCRunResult] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        rows: list[dict[str, float | int | str | None]] = []
        for r in self.results:
            rows.append(
                {
                    "train_size": r.train_size,
                    "seed": r.seed,
                    "test_accuracy": r.test_accuracy,
                    "balanced_accuracy": r.balanced_accuracy,
                    "mcc": r.mcc,
                    "win_rate": r.win_rate,
                    "training_time": r.training_time,
                    "inference_time": r.inference_time,
                    "final_loss": r.final_loss,
                    "ansatz": r.ansatz,
                    "observable": r.observable,
                    "decision_rule": r.decision_rule,
                    "loss_name": r.loss_name,
                }
            )
        return pd.DataFrame(rows)

    def summary(
        self,
        group_cols: Sequence[str] = ("train_size", "ansatz", "loss_name"),
    ) -> pd.DataFrame:
        """Aggregate per-seed metrics as mean/std and bootstrap CI."""
        df = self.to_dataframe()
        if df.empty:
            return df
        metric_cols = [
            "test_accuracy",
            "balanced_accuracy",
            "mcc",
            "win_rate",
            "training_time",
            "inference_time",
        ]
        grouped = df.groupby(list(group_cols), dropna=False)
        rows: list[dict[str, float | int | str]] = []
        for keys, sub in grouped:
            row: dict[str, float | int | str] = {}
            if isinstance(keys, tuple):
                for k, v in zip(group_cols, keys):
                    row[k] = v
            else:
                row[group_cols[0]] = keys
            row["n_runs"] = int(len(sub))
            for m in metric_cols:
                vals = sub[m].dropna().to_numpy(dtype=np.float64)
                if vals.size == 0:
                    row[f"{m}_mean"] = float("nan")
                    row[f"{m}_std"] = float("nan")
                    row[f"{m}_ci_low"] = float("nan")
                    row[f"{m}_ci_high"] = float("nan")
                    continue
                ci_low, ci_high = bootstrap_mean_ci(vals, random_state=42)
                row[f"{m}_mean"] = float(np.mean(vals))
                row[f"{m}_std"] = float(np.std(vals))
                row[f"{m}_ci_low"] = float(ci_low)
                row[f"{m}_ci_high"] = float(ci_high)
            rows.append(row)
        out = pd.DataFrame(rows)
        sort_cols = [c for c in group_cols if c in out.columns]
        return out.sort_values(sort_cols).reset_index(drop=True)

    def statistical_tests(
        self,
        *,
        metrics: Sequence[str] = ("test_accuracy", "win_rate"),
        train_sizes: Sequence[int] | None = None,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """Paired Wilcoxon + effect-size tests across train sizes."""
        df = self.to_dataframe()
        if df.empty:
            return df
        if train_sizes is None:
            train_sizes = sorted(df["train_size"].dropna().unique().tolist())
        frames = [
            sample_efficiency_stat_tests(
                df,
                metric=m,
                train_sizes=train_sizes,
                alpha=alpha,
            )
            for m in metrics
            if m in df.columns
        ]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def power_law_fits(
        self,
        *,
        metrics: Sequence[str] = ("test_accuracy", "win_rate"),
        train_sizes: Sequence[int] | None = None,
    ) -> pd.DataFrame:
        """Fit power-law learning curves for selected metrics."""
        df = self.to_dataframe()
        if df.empty:
            return df
        if train_sizes is None:
            train_sizes = sorted(df["train_size"].dropna().unique().tolist())
        rows: list[dict[str, float | str]] = []
        for metric in metrics:
            if metric not in df.columns:
                continue
            means: list[float] = []
            valid_sizes: list[float] = []
            for size in train_sizes:
                vals = df.loc[df["train_size"] == size, metric].dropna().to_numpy()
                if vals.size == 0:
                    continue
                valid_sizes.append(float(size))
                means.append(float(np.mean(vals)))
            if len(valid_sizes) < 3:
                continue
            fit = fit_power_law_learning_curve(valid_sizes, means)
            rows.append(
                {
                    "metric": metric,
                    **fit,
                }
            )
        return pd.DataFrame(rows)


@dataclass(frozen=True)
class VqcAnsatzHypothesis:
    """Design-time hypothesis for a concrete ansatz choice."""

    ansatz: str
    hypothesis: str
    expected_strength: str
    primary_risk: str


@dataclass
class VqcNoiseSweepRunResult:
    """One VQC run for a noise profile x shot budget x seed."""

    noise_profile: str
    noise_level: float | None
    shots: int
    seed: int
    ansatz: str
    training_time: float
    inference_time: float
    final_loss: float
    test_accuracy_raw: float
    balanced_accuracy_raw: float
    mcc_raw: float
    test_accuracy_readout: float | None = None
    balanced_accuracy_readout: float | None = None
    mcc_readout: float | None = None
    test_accuracy_zne: float | None = None
    balanced_accuracy_zne: float | None = None
    mcc_zne: float | None = None
    test_accuracy_readout_zne: float | None = None
    balanced_accuracy_readout_zne: float | None = None
    mcc_readout_zne: float | None = None


@dataclass
class VqcNoiseSweepResults:
    """Collection of VQC noise-sweep runs."""

    results: list[VqcNoiseSweepRunResult] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        rows: list[dict[str, float | int | str | None]] = []
        for r in self.results:
            rows.append(
                {
                    "noise_profile": r.noise_profile,
                    "noise_level": r.noise_level,
                    "shots": r.shots,
                    "seed": r.seed,
                    "ansatz": r.ansatz,
                    "training_time": r.training_time,
                    "inference_time": r.inference_time,
                    "final_loss": r.final_loss,
                    "test_accuracy_raw": r.test_accuracy_raw,
                    "balanced_accuracy_raw": r.balanced_accuracy_raw,
                    "mcc_raw": r.mcc_raw,
                    "test_accuracy_readout": r.test_accuracy_readout,
                    "balanced_accuracy_readout": r.balanced_accuracy_readout,
                    "mcc_readout": r.mcc_readout,
                    "test_accuracy_zne": r.test_accuracy_zne,
                    "balanced_accuracy_zne": r.balanced_accuracy_zne,
                    "mcc_zne": r.mcc_zne,
                    "test_accuracy_readout_zne": r.test_accuracy_readout_zne,
                    "balanced_accuracy_readout_zne": r.balanced_accuracy_readout_zne,
                    "mcc_readout_zne": r.mcc_readout_zne,
                }
            )
        return pd.DataFrame(rows)

    def summary(
        self,
        group_cols: Sequence[str] = ("noise_profile", "noise_level", "shots"),
    ) -> pd.DataFrame:
        """Aggregate run metrics as mean/std and bootstrap confidence intervals."""
        df = self.to_dataframe()
        if df.empty:
            return df
        metric_cols = [
            "test_accuracy_raw",
            "balanced_accuracy_raw",
            "mcc_raw",
            "test_accuracy_readout",
            "balanced_accuracy_readout",
            "mcc_readout",
            "test_accuracy_zne",
            "balanced_accuracy_zne",
            "mcc_zne",
            "test_accuracy_readout_zne",
            "balanced_accuracy_readout_zne",
            "mcc_readout_zne",
            "training_time",
            "inference_time",
        ]
        grouped = df.groupby(list(group_cols), dropna=False)
        rows: list[dict[str, float | int | str]] = []
        for keys, sub in grouped:
            row: dict[str, float | int | str] = {}
            if isinstance(keys, tuple):
                for k, v in zip(group_cols, keys):
                    row[k] = v
            else:
                row[group_cols[0]] = keys
            row["n_runs"] = int(len(sub))
            for m in metric_cols:
                vals = sub[m].dropna().to_numpy(dtype=np.float64)
                if vals.size == 0:
                    row[f"{m}_mean"] = float("nan")
                    row[f"{m}_std"] = float("nan")
                    row[f"{m}_ci_low"] = float("nan")
                    row[f"{m}_ci_high"] = float("nan")
                    continue
                ci_low, ci_high = bootstrap_mean_ci(vals, random_state=42)
                row[f"{m}_mean"] = float(np.mean(vals))
                row[f"{m}_std"] = float(np.std(vals))
                row[f"{m}_ci_low"] = float(ci_low)
                row[f"{m}_ci_high"] = float(ci_high)
            rows.append(row)
        out = pd.DataFrame(rows)
        sort_cols = [c for c in group_cols if c in out.columns]
        return out.sort_values(sort_cols).reset_index(drop=True)


@dataclass(frozen=True)
class VqcNoiseSweepTask:
    """One task in the VQC noise-design sweep."""

    noise_profile: str
    noise_level: float | None
    shots: int
    seed: int


# ---------------------------------------------------------------------------
# Shot schedule
# ---------------------------------------------------------------------------

DEFAULT_SHOT_SCHEDULE: dict[int, int] = {1: 250, 21: 500, 51: 750}


def shots_for_eval(
    eval_number: int,
    schedule: dict[int, int] | None = None,
) -> int:
    """
    Return the shot count for a given function-evaluation number.

    Default schedule:
      - Evaluations 1–20:  250 shots
      - Evaluations 21–50: 500 shots
      - Evaluations 51+:   750 shots

    Parameters
    ----------
    eval_number : int
        Current function evaluation (1-indexed).
    schedule : dict or None
        Mapping ``{threshold: shots}``.
    """
    if schedule is None:
        schedule = DEFAULT_SHOT_SCHEDULE

    shot_count = 250  # fallback
    for threshold in sorted(schedule.keys()):
        if eval_number >= threshold:
            shot_count = schedule[threshold]
    return shot_count


# ---------------------------------------------------------------------------
# Circuit evaluation
# ---------------------------------------------------------------------------


def evaluate_circuit(
    vc: VariationalClassifier,
    X: np.ndarray,
    theta: np.ndarray,
    shots: int,
    sampler: Any,
) -> np.ndarray:
    """
    Run the parameterised circuit on all samples and return class
    probabilities.

    Parameters
    ----------
    vc : VariationalClassifier
        The circuit (with feature + trainable parameter slots).
    X : ndarray, shape ``(n_samples, n_features)``
        Angle-mapped input features.
    theta : ndarray, shape ``(n_trainable,)``
        Current trainable weights.
    shots : int
        Number of measurement shots per sample.
    sampler
        A Qiskit V2 sampler (``StatevectorSampler`` or Aer ``SamplerV2``).

    Returns
    -------
    ndarray, shape ``(n_samples, n_classes)``
        Class probability matrix.
    """
    outputs = evaluate_circuit_outputs(vc, X, theta, shots, sampler)
    return outputs["class_probs"]


def evaluate_circuit_outputs(
    vc: VariationalClassifier,
    X: np.ndarray,
    theta: np.ndarray,
    shots: int,
    sampler: Any,
    *,
    expectation_qubit: int = 0,
    readout_assignment_matrix: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """
    Run the circuit and return both class probabilities and Z expectations.
    """
    n_samples = X.shape[0]
    bound_values = vc.bind(X, theta)

    ba = BindingsArray({tuple(vc.circuit.parameters): bound_values})
    pub = SamplerPub(circuit=vc.circuit, parameter_values=ba, shots=shots)
    job = sampler.run([pub])
    result = job.result()

    class_probs = np.zeros((n_samples, vc.n_classes), dtype=np.float64)
    z_expectations = np.zeros(n_samples, dtype=np.float64)
    for i in range(n_samples):
        counts = result[0].data.meas.get_counts(i)
        probs = _counts_to_prob_vector(counts, vc.n_qubits)
        if readout_assignment_matrix is not None:
            probs = mitigate_readout_prob_vector(probs, readout_assignment_matrix)
        class_probs[i] = _class_probs_from_bitstring_probs(
            probs, vc.n_classes, vc.class_map
        )
        z_expectations[i] = _z_expectation_from_bitstring_probs(
            probs, qubit=expectation_qubit
        )

    return {"class_probs": class_probs, "z_expectations": z_expectations}


def _expectation_to_p1(z_expectations: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Map <Z> in [-1,1] to p(class=1) in [0,1]."""
    p1 = 0.5 * (1.0 - z_expectations)
    return np.clip(p1, eps, 1.0 - eps)


def _counts_to_prob_vector(
    counts: dict[str, int],
    n_qubits: int,
) -> np.ndarray:
    """Convert counts dictionary to bitstring probability vector."""
    n_states = 2**n_qubits
    probs = np.zeros(n_states, dtype=np.float64)
    total = float(sum(counts.values()))
    if total <= 0:
        return np.ones(n_states, dtype=np.float64) / float(n_states)
    for bitstring, count in counts.items():
        idx = int(bitstring, 2)
        probs[idx] += float(count)
    return probs / total


def _class_probs_from_bitstring_probs(
    bitstring_probs: np.ndarray,
    n_classes: int,
    class_map: dict[int, int],
) -> np.ndarray:
    """Aggregate bitstring probabilities into class probabilities."""
    out = np.zeros(n_classes, dtype=np.float64)
    total = 0.0
    for idx, p in enumerate(bitstring_probs):
        cls = class_map.get(int(idx), -1)
        if cls >= 0:
            out[cls] += float(p)
            total += float(p)
    if total <= 1e-15:
        return np.ones(n_classes, dtype=np.float64) / float(n_classes)
    return out / total


def _z_expectation_from_bitstring_probs(
    bitstring_probs: np.ndarray,
    *,
    qubit: int = 0,
) -> float:
    """Compute <Z_qubit> from full bitstring probabilities."""
    n_states = int(bitstring_probs.shape[0])
    n_qubits = int(np.log2(max(1, n_states)))
    if n_states == 0:
        return 0.0
    if qubit < 0 or qubit >= n_qubits:
        raise ValueError(f"qubit must be in [0, {n_qubits - 1}]")
    z = 0.0
    for idx, p in enumerate(bitstring_probs):
        bit = (idx >> qubit) & 1
        z += float(p) * (1.0 if bit == 0 else -1.0)
    return z


def _loss_from_outputs(
    outputs: dict[str, np.ndarray],
    y_true: np.ndarray,
    *,
    loss_name: LossName,
    eps: float = 1e-10,
) -> float:
    if loss_name == "softmax_nll":
        return batch_loss(outputs["class_probs"], y_true, eps=eps)

    y_true = y_true.astype(np.int64)
    z_expect = outputs["z_expectations"]
    if loss_name == "cross_entropy_expectation":
        p1 = _expectation_to_p1(z_expect, eps=eps)
        losses = -(y_true * np.log(p1) + (1 - y_true) * np.log(1.0 - p1))
        return float(np.mean(losses))

    # Binary hinge loss on margin score s(x) = -<Z>; y in {-1, +1}
    y_pm = 2 * y_true - 1
    score = -z_expect
    margins = y_pm * score
    return float(np.mean(np.maximum(0.0, 1.0 - margins)))


def _predict_from_outputs(
    outputs: dict[str, np.ndarray],
    *,
    decision_rule: DecisionRule,
) -> np.ndarray:
    if decision_rule == "argmax":
        return predict_batch(outputs["class_probs"])
    return (outputs["z_expectations"] < 0.0).astype(np.int64)


def vqc_policy(
    vc: VariationalClassifier,
    theta: np.ndarray,
    feature_fn: Callable[[np.ndarray], np.ndarray],
    *,
    shots: int = 300,
    sampler: Any | None = None,
    seed: int = 42,
    decision_rule: DecisionRule = "argmax",
    expectation_qubit: int = 0,
) -> Policy:
    """Wrap a trained VQC as a Nim move policy."""
    policy_sampler = sampler if sampler is not None else StatevectorSampler(seed=seed)

    def policy(state: NimState, rng: np.random.Generator) -> NimMove:
        moves = legal_moves(state)
        if len(moves) == 1:
            return moves[0]

        resulting_states = np.array(
            [apply_move(state, m) for m in moves],
            dtype=np.int32,
        )
        X = feature_fn(resulting_states)
        outputs = evaluate_circuit_outputs(
            vc,
            X,
            theta,
            shots,
            policy_sampler,
            expectation_qubit=expectation_qubit,
        )
        preds = _predict_from_outputs(outputs, decision_rule=decision_rule)
        good_idx = np.flatnonzero(preds == 0)
        if good_idx.size > 0:
            return moves[int(rng.choice(good_idx))]
        return moves[int(rng.integers(len(moves)))]

    return policy


def evaluate_vqc_win_rate(
    vc: VariationalClassifier,
    theta: np.ndarray,
    feature_fn: Callable[[np.ndarray], np.ndarray],
    *,
    n_games: int = 200,
    k: int = 3,
    M: int = 7,
    seed: int = 42,
    shots: int = 300,
    sampler: Any | None = None,
    decision_rule: DecisionRule = "argmax",
    expectation_qubit: int = 0,
) -> float:
    """Play VQC policy vs random and return first-player win rate."""
    pol = vqc_policy(
        vc,
        theta,
        feature_fn,
        shots=shots,
        sampler=sampler,
        seed=seed,
        decision_rule=decision_rule,
        expectation_qubit=expectation_qubit,
    )
    stats = play_many(pol, random_policy, n_games=n_games, k=k, M=M, seed=seed)
    return float(stats["win_rate_a"])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Single-seed training
# ---------------------------------------------------------------------------


def train_classifier(
    vc: VariationalClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    *,
    max_iter: int = 200,
    shot_schedule: dict[int, int] | None = None,
    seed: int = 42,
    test_shots: int = 300,
    sampler: Any | None = None,
    observable: MeasurementObservable = "bitstring_probs",
    decision_rule: DecisionRule = "argmax",
    loss_name: LossName = "softmax_nll",
    expectation_qubit: int = 0,
    verbose: bool = True,
    log_interval: int = 10,
    mlflow_experiment: str | None = None,
) -> tuple[np.ndarray, TrainingHistory]:
    """
    Train the variational classifier using COBYLA optimisation.

    Uses gradient-free COBYLA optimisation, progressive shot schedule, and
    random initialisation in :math:`[-\\pi, \\pi]`.

    Parameters
    ----------
    vc : VariationalClassifier
        Circuit to train.
    X_train, y_train : ndarray
        Training data (angle-mapped) and labels.
    X_test, y_test : ndarray or None
        Optional test data for tracking generalisation during training.
    max_iter : int
        Maximum number of COBYLA iterations (default 200).
    shot_schedule : dict or None
        Mapping ``{eval_number: shots}``.  Default: 250/500/750.
    seed : int
        RNG seed for weight initialisation *and* sampler reproducibility.
    test_shots : int
        Shots per sample for test evaluation (default 300).
    sampler
        Sampler to use.  If *None*, a ``StatevectorSampler(seed=seed)``
        is created (ideal simulation).  Pass a noisy sampler for noise
        experiments.
    observable : ``"bitstring_probs"`` | ``"z_expectation"``
        Measurement observable used for reporting; expectation is also used
        when the selected loss or decision rule depends on it.
    decision_rule : ``"argmax"`` | ``"expectation_threshold"``
        Prediction mapping from sampler outputs to classes.
    loss_name : ``"softmax_nll"`` | ``"cross_entropy_expectation"`` |
        ``"hinge_expectation"``
        Objective function used by COBYLA.
    expectation_qubit : int
        Qubit index for computing :math:`\\langle Z \\rangle`.
    verbose : bool
        Print progress every *log_interval* evaluations.
    log_interval : int
        How often to record and print metrics (in function evaluations).

    Returns
    -------
    (best_weights, history) : tuple[ndarray, TrainingHistory]
    """
    rng = np.random.default_rng(seed)
    theta_init = rng.uniform(-np.pi, np.pi, vc.n_trainable)

    if sampler is None:
        sampler = StatevectorSampler(seed=seed)
    if loss_name != "softmax_nll" and vc.n_classes != 2:
        raise ValueError("Expectation-based losses require binary classes.")
    if decision_rule == "expectation_threshold" and vc.n_classes != 2:
        raise ValueError("Expectation-threshold decision rule requires binary classes.")
    if observable not in ("bitstring_probs", "z_expectation"):
        raise ValueError("observable must be 'bitstring_probs' or 'z_expectation'")

    history = TrainingHistory()
    eval_counter = [0]  # mutable for closure access

    # MLflow logging setup
    mlflow_run = None
    if mlflow_experiment:
        try:
            import mlflow
            mlflow.set_experiment(mlflow_experiment)
            mlflow_run = mlflow.start_run()
            mlflow.log_params({
                "seed": seed,
                "max_iter": max_iter,
                "n_qubits": vc.n_qubits,
                "n_features": vc.n_features,
                "n_classes": vc.n_classes,
                "n_trainable": vc.n_trainable,
                "ansatz": vc.ansatz,
                "test_shots": test_shots,
                "observable": observable,
                "decision_rule": decision_rule,
                "loss_name": loss_name,
                "expectation_qubit": expectation_qubit,
            })
        except ImportError:
            if verbose:
                print("Warning: MLflow not available, skipping logging")
            mlflow_run = None

    t0 = time.perf_counter()

    def objective(params: np.ndarray) -> float:
        eval_counter[0] += 1
        n_eval = eval_counter[0]
        shots = shots_for_eval(n_eval, shot_schedule)

        outputs = evaluate_circuit_outputs(
            vc,
            X_train,
            params,
            shots,
            sampler,
            expectation_qubit=expectation_qubit,
        )
        loss_val = _loss_from_outputs(outputs, y_train, loss_name=loss_name)

        # Track best
        if loss_val < history.best_loss:
            history.best_loss = loss_val
            history.best_weights = params.copy()

        # Log at intervals
        if n_eval % log_interval == 0 or n_eval == 1:
            train_preds = _predict_from_outputs(outputs, decision_rule=decision_rule)
            train_acc = float(np.mean(train_preds == y_train))

            history.losses.append(loss_val)
            history.train_accuracies.append(train_acc)
            history.eval_numbers.append(n_eval)
            history.shot_counts.append(shots)

            if X_test is not None and y_test is not None:
                test_outputs = evaluate_circuit_outputs(
                    vc,
                    X_test,
                    params,
                    test_shots,
                    sampler,
                    expectation_qubit=expectation_qubit,
                )
                test_preds = _predict_from_outputs(
                    test_outputs, decision_rule=decision_rule
                )
                test_acc = float(np.mean(test_preds == y_test))
                history.test_accuracies.append(test_acc)

            if verbose:
                msg = (
                    f"  Eval {n_eval:4d} | loss={loss_val:.4f}"
                    f" | train_acc={train_acc:.3f}"
                )
                if history.test_accuracies:
                    msg += f" | test_acc={history.test_accuracies[-1]:.3f}"
                msg += f" | shots={shots}"
                print(msg)

        return loss_val

    # Run COBYLA
    opt_result = minimize(
        objective,
        theta_init,
        method="COBYLA",
        options={"maxiter": max_iter, "rhobeg": 0.5},
    )

    history.total_training_time = time.perf_counter() - t0
    history.total_evals = eval_counter[0]

    best_weights: np.ndarray = (
        history.best_weights
        if history.best_weights is not None
        else opt_result.x
    )
    history.best_weights = best_weights

    # MLflow logging of final metrics
    if mlflow_run:
        try:
            import mlflow
            mlflow.log_metrics({
                "train_accuracy": history.train_accuracies[-1] if history.train_accuracies else 0.0,
                "test_accuracy": history.test_accuracies[-1] if history.test_accuracies else 0.0,
                "final_loss": history.best_loss,
                "training_time": history.total_training_time,
                "total_evals": history.total_evals,
            })
            mlflow.end_run()
        except Exception as e:
            if verbose:
                print(f"Warning: MLflow logging failed: {e}")

    if verbose:
        print(
            f"\nTraining complete in {history.total_training_time:.1f}s "
            f"({history.total_evals} evaluations)"
        )
        print(f"Best loss: {history.best_loss:.4f}")
        print(f"COBYLA status: {opt_result.message}")

    return best_weights, history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_classifier(
    vc: VariationalClassifier,
    X: np.ndarray,
    y: np.ndarray,
    theta: np.ndarray,
    *,
    shots: int = 300,
    sampler: Any | None = None,
    seed: int = 42,
    decision_rule: DecisionRule = "argmax",
    expectation_qubit: int = 0,
) -> dict:
    """
    Evaluate a trained classifier on held-out data.

    Returns
    -------
    dict
        Keys: ``accuracy``, ``predictions``, ``class_probs``,
        ``inference_time``.
    """
    if sampler is None:
        sampler = StatevectorSampler(seed=seed)

    t0 = time.perf_counter()
    outputs = evaluate_circuit_outputs(
        vc,
        X,
        theta,
        shots,
        sampler,
        expectation_qubit=expectation_qubit,
    )
    inference_time = time.perf_counter() - t0

    preds = _predict_from_outputs(outputs, decision_rule=decision_rule)
    accuracy = float(np.mean(preds == y))

    return {
        "accuracy": accuracy,
        "predictions": preds,
        "class_probs": outputs["class_probs"],
        "z_expectations": outputs["z_expectations"],
        "inference_time": inference_time,
    }


# ---------------------------------------------------------------------------
# Multi-seed experiment
# ---------------------------------------------------------------------------


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
    force_rerun: bool = False,
) -> MultiSeedSummary:
    """
    Train with multiple random seeds and aggregate results.

    This directly addresses QML model volatility: models can be quite
    volatile and sensitive to starting conditions, so multiple seeds
    are recommended.

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
        load matching finished runs from MLflow instead of training.
    force_rerun : bool
        If True, always train and log; ignore MLflow cache.
    """
    if seeds is None:
        seeds = list(range(n_seeds))

    temp_vc = vc_builder()

    if (
        use_cache
        and not force_rerun
        and mlflow_experiment
        and mlflow_run_name
    ):
        cached = _load_multi_seed_summary_from_mlflow(
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
        if cached is not None:
            return cached

    # MLflow parent run setup
    mlflow_parent_run = None
    if mlflow_experiment:
        try:
            import mlflow

            _set_mlflow_tracking_uri()
            mlflow.set_experiment(mlflow_experiment)
            mlflow_parent_run = mlflow.start_run(run_name=mlflow_run_name)
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

    all_results: list[ExperimentResult] = []

    for i, seed in enumerate(seeds):
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Seed {seed} ({i + 1}/{len(seeds)})")
            print(f"{'=' * 60}")

        # MLflow nested run for this seed
        if mlflow_parent_run:
            try:
                import mlflow
                mlflow.start_run(run_name=f"seed_{seed}", nested=True)
            except Exception:
                pass

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
            mlflow_experiment=None,  # Don't double-log
        )

        # Final evaluation with the best weights
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

        # Log seed-specific metrics to nested run
        if mlflow_parent_run:
            try:
                import mlflow

                bal_acc = balanced_accuracy_score(
                    y_test, eval_result["predictions"]
                )
                mcc_val = matthews_corrcoef(
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
                mlflow.end_run()  # End nested run
            except Exception:
                pass

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

    # Aggregate
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

    # Log aggregate metrics to parent run
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
            mlflow.end_run()  # End parent run
        except Exception as e:
            if verbose:
                print(f"Warning: MLflow logging failed: {e}")

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


def bootstrap_mean_ci(
    values: np.ndarray | Sequence[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    random_state: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean."""
    vals = np.asarray(values, dtype=np.float64)
    if vals.size == 0:
        return (float("nan"), float("nan"))
    if vals.size == 1:
        v = float(vals[0])
        return (v, v)

    rng = np.random.default_rng(random_state)
    idx = rng.integers(0, vals.size, size=(n_resamples, vals.size))
    means = vals[idx].mean(axis=1)
    alpha = 1.0 - confidence
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return (lo, hi)


def paired_cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d for paired samples based on within-seed deltas."""
    diff = np.asarray(y, dtype=np.float64) - np.asarray(x, dtype=np.float64)
    if diff.size < 2:
        return 0.0
    sd = float(np.std(diff, ddof=1))
    if np.isclose(sd, 0.0):
        return 0.0
    return float(np.mean(diff) / sd)


def rank_biserial_from_deltas(deltas: np.ndarray) -> float:
    """Rank-biserial sign effect size from paired deltas."""
    d = np.asarray(deltas, dtype=np.float64)
    nonzero = d[np.abs(d) > 1e-15]
    if nonzero.size == 0:
        return 0.0
    n_pos = int(np.sum(nonzero > 0))
    n_neg = int(np.sum(nonzero < 0))
    return float((n_pos - n_neg) / (n_pos + n_neg))


def sample_efficiency_stat_tests(
    df: pd.DataFrame,
    *,
    metric: str,
    train_sizes: Sequence[int],
    seed_col: str = "seed",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Pairwise train-size significance tests for one metric.

    Uses paired Wilcoxon signed-rank on common seeds with Bonferroni correction.
    """
    if metric not in df.columns:
        return pd.DataFrame()

    try:
        from scipy.stats import wilcoxon
    except Exception:
        return pd.DataFrame()

    if len(train_sizes) < 2:
        return pd.DataFrame()

    rows: list[dict[str, float | int | str | bool]] = []
    m_tests = max(1, len(list(itertools.combinations(train_sizes, 2))))
    for a, b in itertools.combinations(train_sizes, 2):
        sa = (
            df.loc[df["train_size"] == a, [seed_col, metric]]
            .dropna()
            .drop_duplicates(subset=[seed_col])
            .set_index(seed_col)[metric]
        )
        sb = (
            df.loc[df["train_size"] == b, [seed_col, metric]]
            .dropna()
            .drop_duplicates(subset=[seed_col])
            .set_index(seed_col)[metric]
        )
        common = sa.index.intersection(sb.index)
        if common.empty:
            continue
        x = sa.loc[common].to_numpy(dtype=np.float64)
        y = sb.loc[common].to_numpy(dtype=np.float64)
        if x.size == 0:
            continue
        if np.allclose(x, y):
            stat, p_val = 0.0, 1.0
        else:
            stat, p_val = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
        p_corr = min(1.0, float(p_val) * m_tests)
        deltas = y - x
        rows.append(
            {
                "metric": metric,
                "size_a": int(a),
                "size_b": int(b),
                "n_pairs": int(x.size),
                "mean_a": float(np.mean(x)),
                "std_a": float(np.std(x)),
                "mean_b": float(np.mean(y)),
                "std_b": float(np.std(y)),
                "mean_delta_b_minus_a": float(np.mean(deltas)),
                "wilcoxon_stat": float(stat),
                "p_value": float(p_val),
                "p_value_bonferroni": float(p_corr),
                "reject_null_alpha": bool(p_corr < alpha),
                "cohens_d_paired": float(paired_cohens_d(x, y)),
                "rank_biserial": float(rank_biserial_from_deltas(deltas)),
            }
        )
    return pd.DataFrame(rows)


def fit_power_law_learning_curve(
    train_sizes: Sequence[float],
    metric_values: Sequence[float],
) -> dict[str, float]:
    """Fit accuracy = a - b * n^(-c) and return fit diagnostics."""
    x = np.asarray(train_sizes, dtype=np.float64)
    y = np.asarray(metric_values, dtype=np.float64)
    if x.size < 3 or y.size < 3 or x.size != y.size:
        return {
            "a": float("nan"),
            "b": float("nan"),
            "c": float("nan"),
            "r2": float("nan"),
        }

    def model(n: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        return a - b * np.power(n, -c)

    try:
        from scipy.optimize import OptimizeWarning, curve_fit

        p0 = [float(np.max(y)), 0.2, 0.5]
        bounds = ([0.0, 0.0, 1e-6], [2.0, 10.0, 10.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            params, _ = curve_fit(
                model, x, y, p0=p0, bounds=bounds, maxfev=50_000
            )
        y_hat = model(x, *params)
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = float("nan") if np.isclose(ss_tot, 0.0) else 1.0 - ss_res / ss_tot
        return {
            "a": float(params[0]),
            "b": float(params[1]),
            "c": float(params[2]),
            "r2": float(r2),
        }
    except Exception:
        return {
            "a": float("nan"),
            "b": float("nan"),
            "c": float("nan"),
            "r2": float("nan"),
        }


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


_vqc_ood_pool: dict[str, Any] = {}


def _vqc_ood_pool_init(cfg: dict[str, Any]) -> None:
    """Worker initializer: one picklable dict (ProcessPoolExecutor initargs)."""
    _vqc_ood_pool.clear()
    _vqc_ood_pool.update(cfg)


def simulated_vqc_ood_single_run(
    vc: VariationalClassifier,
    subset_X: np.ndarray,
    subset_y: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    train_size: int,
    seed: int,
    max_iter: int,
    shot_schedule: dict[int, int] | None,
    test_shots: int,
    sampler: Any | None,
    decision_rule: DecisionRule,
    observable: MeasurementObservable,
    loss_name: LossName,
    expectation_qubit: int,
    feature_fn_for_policy: Callable[[np.ndarray], np.ndarray] | None,
    compute_win_rate: bool,
    n_games_win_rate: int,
    game_k: int,
    game_M: int,
    train_verbose: bool,
    log_interval: int,
) -> SimulatedVQCRunResult:
    """Train and evaluate one VQC on a train subset (OOD protocol)."""
    best_weights, history = train_classifier(
        vc,
        subset_X,
        subset_y,
        X_test,
        y_test,
        max_iter=max_iter,
        shot_schedule=shot_schedule,
        seed=int(seed),
        test_shots=test_shots,
        sampler=sampler,
        observable=observable,
        decision_rule=decision_rule,
        loss_name=loss_name,
        expectation_qubit=expectation_qubit,
        verbose=train_verbose,
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
        seed=int(seed),
        decision_rule=decision_rule,
        expectation_qubit=expectation_qubit,
    )
    bal_acc = float(balanced_accuracy_score(y_test, eval_result["predictions"]))
    mcc_val = float(matthews_corrcoef(y_test, eval_result["predictions"]))
    win_rate_val: float | None = None
    if compute_win_rate and feature_fn_for_policy is not None:
        win_rate_val = evaluate_vqc_win_rate(
            vc,
            best_weights,
            feature_fn_for_policy,
            n_games=n_games_win_rate,
            k=game_k,
            M=game_M,
            seed=int(seed),
            shots=test_shots,
            sampler=sampler,
            decision_rule=decision_rule,
            expectation_qubit=expectation_qubit,
        )
    return SimulatedVQCRunResult(
        train_size=int(train_size),
        seed=int(seed),
        test_accuracy=float(eval_result["accuracy"]),
        balanced_accuracy=bal_acc,
        mcc=mcc_val,
        win_rate=win_rate_val,
        training_time=float(history.total_training_time),
        inference_time=float(eval_result["inference_time"]),
        final_loss=float(history.best_loss),
        ansatz=str(vc.ansatz),
        observable=observable,
        decision_rule=decision_rule,
        loss_name=loss_name,
    )


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


def _make_vqc_factory(
    vc_builder: Callable[[], VariationalClassifier] | None,
    circuit_kwargs: dict[str, Any] | None,
) -> Callable[[], VariationalClassifier]:
    if circuit_kwargs is not None and vc_builder is not None:
        raise ValueError("Pass at most one of vc_builder and circuit_kwargs.")
    if circuit_kwargs is not None:

        def factory_ck() -> VariationalClassifier:
            return build_circuit(**circuit_kwargs)

        return factory_ck
    if vc_builder is None:
        raise ValueError("Either vc_builder or circuit_kwargs is required.")
    return vc_builder


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
    use_cache: bool = True,
    force_rerun: bool = False,
    verbose: bool = True,
    max_workers: int | None = None,
    use_tqdm: bool = True,
    log_interval: int = 20,
) -> SimulatedVQCSweepResults:
    """
    Run OOD VQC training at multiple train sizes and seeds.

    The caller should pass OOD arrays (train on M<=5, test on M>5) and encoded
    features. Train-size subsets are stratified per seed.

    When ``use_cache=True`` and ``mlflow_experiment`` is set, finished runs
    matching the grid (including ``mlflow_run_prefix`` in the run name) are
    loaded from MLflow; only missing points are trained. Use ``force_rerun=True``
    or ``use_cache=False`` to ignore the cache.

    Provide exactly one of ``vc_builder`` or ``circuit_kwargs``. For
    ``max_workers`` > 1, use ``circuit_kwargs`` (picklable) and omit
    ``sampler_factory``; MLflow is logged from the parent process.
    """
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

            _set_mlflow_tracking_uri()
            mlflow.set_experiment(mlflow_experiment)
            mlflow_available = True
        except ImportError:
            mlflow_available = False
            if verbose:
                print("Warning: MLflow not available; sweep runs will not be logged.")

    full_train_size = int(len(X_train))
    temp_vc = factory()
    vqc_cache: dict[tuple[int, int], SimulatedVQCRunResult] = {}
    if (
        use_cache
        and not force_rerun
        and mlflow_available
        and mlflow_experiment
    ):
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
                idx += 1
                if verbose and (max_workers is None or max_workers <= 1):
                    print(
                        f"[sim-vqc {run_idx}/{total_runs}] seed={seed} "
                        f"train_size={size} (cached)"
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

    computed: list[SimulatedVQCRunResult] = []
    if pending:
        tasks_only = [t for _, t in pending]
        if max_workers is None or max_workers <= 1:
            serial_results: list[SimulatedVQCRunResult] = []
            task_iter = tasks_only
            if use_tqdm:
                from tqdm.auto import tqdm as _tqdm

                task_iter = _tqdm(
                    tasks_only,
                    desc="simulated VQC OOD",
                    total=len(tasks_only),
                )
            for task in task_iter:
                vc = factory()
                sampler = (
                    sampler_factory(int(task.seed))
                    if sampler_factory is not None
                    else None
                )
                serial_results.append(
                    simulated_vqc_ood_single_run(
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
                )
            computed = serial_results
        else:
            assert circuit_kwargs is not None
            pool_cfg: dict[str, Any] = {
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
            computed = map_parallel_or_serial(
                tasks_only,
                _vqc_ood_worker,
                max_workers=max_workers,
                use_tqdm=use_tqdm,
                tqdm_desc="simulated VQC OOD",
                initializer=_vqc_ood_pool_init,
                initargs=(pool_cfg,),
            )

        for (i, _), res in zip(pending, computed, strict=True):
            ordered[i] = res

    sweep = SimulatedVQCSweepResults()
    sweep.results = [r for r in ordered if r is not None]

    if mlflow_available and computed:
        import mlflow

        vc_meta = temp_vc
        for res in computed:
            run_name = f"{mlflow_run_prefix}|n={res.train_size}|s={res.seed}"
            with mlflow.start_run(run_name=run_name):
                mlflow.log_params(
                    {
                        "pipeline": "simulated_vqc",
                        "regime": "ood",
                        "train_size": res.train_size,
                        "seed": int(res.seed),
                        "max_iter": max_iter,
                        "test_shots": test_shots,
                        "ansatz": res.ansatz,
                        "n_qubits": vc_meta.n_qubits,
                        "n_features": vc_meta.n_features,
                        "n_trainable": vc_meta.n_trainable,
                        "observable": observable,
                        "decision_rule": decision_rule,
                        "loss_name": loss_name,
                        "expectation_qubit": expectation_qubit,
                        "n_games_win_rate": n_games_win_rate,
                    }
                )
                metrics: dict[str, float] = {
                    "test_accuracy": float(res.test_accuracy),
                    "balanced_accuracy": float(res.balanced_accuracy),
                    "mcc": float(res.mcc),
                    "training_time": float(res.training_time),
                    "inference_time": float(res.inference_time),
                    "final_loss": float(res.final_loss),
                }
                if res.win_rate is not None:
                    metrics["win_rate"] = float(res.win_rate)
                mlflow.log_metrics(metrics)

    return sweep


# ---------------------------------------------------------------------------
# Noise model helpers (requires qiskit-aer)
# ---------------------------------------------------------------------------


def create_depolarizing_noise_model(
    cz_error_rate: float = 0.01,
    single_gate_error_rate: float = 0.0,
    readout_error_rate: float = 0.0,
) -> Any:
    """
    Create a depolarizing noise model for simulation.

    Parameters
    ----------
    cz_error_rate : float
        Depolarizing error probability on CZ (two-qubit) gates.
    single_gate_error_rate : float
        Depolarizing error probability on single-qubit gates (rx, rz).

    Returns
    -------
    qiskit_aer.noise.NoiseModel
    """
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    noise_model = NoiseModel()

    if cz_error_rate > 0:
        error_cz = depolarizing_error(cz_error_rate, 2)
        noise_model.add_all_qubit_quantum_error(error_cz, ["cz"])

    if single_gate_error_rate > 0:
        error_1q = depolarizing_error(single_gate_error_rate, 1)
        noise_model.add_all_qubit_quantum_error(error_1q, ["rx", "rz"])

    if readout_error_rate > 0:
        from qiskit_aer.noise import ReadoutError

        p = float(readout_error_rate)
        ro = ReadoutError([[1.0 - p, p], [p, 1.0 - p]])
        noise_model.add_all_qubit_readout_error(ro)

    return noise_model


def create_noisy_sampler(
    noise_model: Any,
    seed: int = 42,
) -> Any:
    """
    Create a V2 sampler backed by Qiskit Aer with a noise model.

    Parameters
    ----------
    noise_model
        A ``qiskit_aer.noise.NoiseModel``.
    seed : int
        Simulator random seed for reproducibility.

    Returns
    -------
    A sampler compatible with ``StatevectorSampler``'s V2 interface.
    """
    from qiskit_aer.primitives import SamplerV2

    return SamplerV2(
        options={
            "backend_options": {
                "noise_model": noise_model,
                "seed_simulator": seed,
            },
        },
        seed=seed,
    )


def default_vqc_ansatz_hypotheses() -> dict[str, VqcAnsatzHypothesis]:
    """
    Return explicit, report-ready hypotheses for built-in ansatze.

    These hypotheses are used by notebook narrative and MLflow metadata.
    """
    return {
        "basic_block": VqcAnsatzHypothesis(
            ansatz="basic_block",
            hypothesis=(
                "RX(pi/2)-RZ-RX(pi/2) layers bias each qubit toward phase rotations "
                "that can represent parity-sensitive boundaries with modest depth."
            ),
            expected_strength=(
                "Higher expressivity at the same depth; stronger ceiling on clean simulation."
            ),
            primary_risk=(
                "May become noise-sensitive because each block has extra 1-qubit gates."
            ),
        ),
        "ry_rz": VqcAnsatzHypothesis(
            ansatz="ry_rz",
            hypothesis=(
                "Lean RY-RZ blocks reduce gate count so training under noise remains "
                "more stable, even if clean-sim expressivity is slightly lower."
            ),
            expected_strength=(
                "Better robustness under depolarising/readout noise at fixed shots."
            ),
            primary_risk=(
                "Potential underfitting if shallow depth cannot represent full Nim-sum geometry."
            ),
        ),
    }


def build_assignment_matrix_from_symmetric_readout_error(
    *,
    n_qubits: int,
    readout_error_rate: float,
) -> np.ndarray:
    """Construct full assignment matrix A where p_obs = A @ p_true."""
    if n_qubits < 1:
        raise ValueError("n_qubits must be >= 1")
    p = float(readout_error_rate)
    if p < 0.0 or p >= 0.5:
        raise ValueError("readout_error_rate must be in [0, 0.5).")
    one_q = np.array([[1.0 - p, p], [p, 1.0 - p]], dtype=np.float64)
    mat = one_q
    for _ in range(n_qubits - 1):
        mat = np.kron(one_q, mat)
    return mat


def mitigate_readout_prob_vector(
    observed_probs: np.ndarray,
    assignment_matrix: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Apply linear-inversion readout correction with clipping and renormalisation.
    """
    obs = np.asarray(observed_probs, dtype=np.float64)
    A = np.asarray(assignment_matrix, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("assignment_matrix must be square.")
    if obs.ndim != 1 or obs.shape[0] != A.shape[0]:
        raise ValueError("observed_probs shape must match assignment_matrix.")
    corrected = np.linalg.pinv(A) @ obs
    corrected = np.clip(corrected, 0.0, None)
    total = float(np.sum(corrected))
    if total <= eps:
        return np.ones_like(corrected) / float(corrected.size)
    return corrected / total


def zne_extrapolate_to_zero(
    scales: Sequence[float],
    values: Sequence[float],
    *,
    degree: int = 1,
) -> float:
    """Extrapolate scalar metric from scaled-noise values to zero noise."""
    x = np.asarray(scales, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if x.size != y.size or x.size == 0:
        raise ValueError("scales and values must have the same non-zero length.")
    deg = int(min(max(1, degree), x.size - 1))
    coeff = np.polyfit(x, y, deg=deg)
    return float(np.polyval(coeff, 0.0))


def _zne_extrapolate_outputs(
    outputs_per_scale: Sequence[dict[str, np.ndarray]],
    *,
    scales: Sequence[float],
    degree: int,
) -> dict[str, np.ndarray]:
    """Extrapolate class probabilities and Z expectations to zero noise."""
    if len(outputs_per_scale) == 0:
        raise ValueError("outputs_per_scale cannot be empty.")
    x = np.asarray(scales, dtype=np.float64)
    cp_stack = np.stack([o["class_probs"] for o in outputs_per_scale], axis=0)
    z_stack = np.stack([o["z_expectations"] for o in outputs_per_scale], axis=0)
    deg = int(min(max(1, degree), len(outputs_per_scale) - 1))

    n_samples, n_classes = cp_stack.shape[1], cp_stack.shape[2]
    cp0 = np.zeros((n_samples, n_classes), dtype=np.float64)
    for i in range(n_samples):
        for j in range(n_classes):
            coeff = np.polyfit(x, cp_stack[:, i, j], deg=deg)
            cp0[i, j] = float(np.polyval(coeff, 0.0))
    cp0 = np.clip(cp0, 0.0, None)
    cp0_sum = cp0.sum(axis=1, keepdims=True)
    cp0_sum = np.where(cp0_sum <= 1e-12, 1.0, cp0_sum)
    cp0 = cp0 / cp0_sum

    z0 = np.zeros(n_samples, dtype=np.float64)
    for i in range(n_samples):
        coeff = np.polyfit(x, z_stack[:, i], deg=deg)
        z0[i] = float(np.polyval(coeff, 0.0))
    z0 = np.clip(z0, -1.0, 1.0)

    return {"class_probs": cp0, "z_expectations": z0}


def _metrics_from_preds(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """Return (accuracy, balanced_accuracy, MCC)."""
    acc = float(np.mean(y_pred == y_true))
    bal = float(balanced_accuracy_score(y_true, y_pred))
    mcc_val = float(matthews_corrcoef(y_true, y_pred))
    return acc, bal, mcc_val


_vqc_noise_pool: dict[str, Any] = {}


def _vqc_noise_pool_init(cfg: dict[str, Any]) -> None:
    _vqc_noise_pool.clear()
    _vqc_noise_pool.update(cfg)


def _single_noise_run(
    *,
    vc: VariationalClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task: VqcNoiseSweepTask,
    max_iter: int,
    shot_schedule: dict[int, int] | None,
    decision_rule: DecisionRule,
    observable: MeasurementObservable,
    loss_name: LossName,
    expectation_qubit: int,
    zne_scales: Sequence[float],
    zne_degree: int,
    single_gate_error_ratio: float,
    readout_error_rate: float,
    backend_noise_model: Any | None,
    apply_readout_correction: bool,
    apply_zne: bool,
    log_interval: int,
) -> VqcNoiseSweepRunResult:
    """Train/evaluate one point of the noise sweep."""
    seed = int(task.seed)
    if task.noise_profile == "depolarizing":
        base_rate = float(task.noise_level or 0.0)
        noise_model = create_depolarizing_noise_model(
            cz_error_rate=base_rate,
            single_gate_error_rate=single_gate_error_ratio * base_rate,
            readout_error_rate=readout_error_rate,
        )
    elif task.noise_profile == "backend":
        if backend_noise_model is None:
            raise ValueError("backend_noise_model is required for backend profile.")
        noise_model = backend_noise_model
    else:
        raise ValueError(f"Unsupported noise profile: {task.noise_profile}")

    sampler = create_noisy_sampler(noise_model, seed=seed)
    best_weights, history = train_classifier(
        vc,
        X_train,
        y_train,
        X_test,
        y_test,
        max_iter=max_iter,
        shot_schedule=shot_schedule,
        seed=seed,
        test_shots=int(task.shots),
        sampler=sampler,
        observable=observable,
        decision_rule=decision_rule,
        loss_name=loss_name,
        expectation_qubit=expectation_qubit,
        verbose=False,
        log_interval=log_interval,
        mlflow_experiment=None,
    )

    t0 = time.perf_counter()
    base_outputs = evaluate_circuit_outputs(
        vc,
        X_test,
        best_weights,
        int(task.shots),
        sampler,
        expectation_qubit=expectation_qubit,
    )
    inference_time = time.perf_counter() - t0
    raw_preds = _predict_from_outputs(base_outputs, decision_rule=decision_rule)
    acc_raw, bal_raw, mcc_raw = _metrics_from_preds(y_test, raw_preds)

    readout_matrix: np.ndarray | None = None
    if apply_readout_correction:
        if task.noise_profile == "depolarizing" and readout_error_rate > 0:
            readout_matrix = build_assignment_matrix_from_symmetric_readout_error(
                n_qubits=vc.n_qubits,
                readout_error_rate=readout_error_rate,
            )

    readout_metrics: tuple[float, float, float] | None = None
    if readout_matrix is not None:
        readout_outputs = evaluate_circuit_outputs(
            vc,
            X_test,
            best_weights,
            int(task.shots),
            sampler,
            expectation_qubit=expectation_qubit,
            readout_assignment_matrix=readout_matrix,
        )
        readout_preds = _predict_from_outputs(readout_outputs, decision_rule=decision_rule)
        readout_metrics = _metrics_from_preds(y_test, readout_preds)

    zne_metrics: tuple[float, float, float] | None = None
    readout_zne_metrics: tuple[float, float, float] | None = None
    if apply_zne and task.noise_profile == "depolarizing":
        outputs_by_scale: list[dict[str, np.ndarray]] = []
        outputs_by_scale_readout: list[dict[str, np.ndarray]] = []
        for scale in zne_scales:
            scaled_rate = min(float(task.noise_level or 0.0) * float(scale), 0.49)
            scaled_noise = create_depolarizing_noise_model(
                cz_error_rate=scaled_rate,
                single_gate_error_rate=single_gate_error_ratio * scaled_rate,
                readout_error_rate=readout_error_rate,
            )
            scaled_sampler = create_noisy_sampler(scaled_noise, seed=seed)
            outputs_by_scale.append(
                evaluate_circuit_outputs(
                    vc,
                    X_test,
                    best_weights,
                    int(task.shots),
                    scaled_sampler,
                    expectation_qubit=expectation_qubit,
                )
            )
            if readout_matrix is not None:
                outputs_by_scale_readout.append(
                    evaluate_circuit_outputs(
                        vc,
                        X_test,
                        best_weights,
                        int(task.shots),
                        scaled_sampler,
                        expectation_qubit=expectation_qubit,
                        readout_assignment_matrix=readout_matrix,
                    )
                )

        zne_outputs = _zne_extrapolate_outputs(
            outputs_by_scale,
            scales=zne_scales,
            degree=zne_degree,
        )
        zne_preds = _predict_from_outputs(zne_outputs, decision_rule=decision_rule)
        zne_metrics = _metrics_from_preds(y_test, zne_preds)

        if outputs_by_scale_readout:
            zne_outputs_readout = _zne_extrapolate_outputs(
                outputs_by_scale_readout,
                scales=zne_scales,
                degree=zne_degree,
            )
            zne_preds_readout = _predict_from_outputs(
                zne_outputs_readout, decision_rule=decision_rule
            )
            readout_zne_metrics = _metrics_from_preds(y_test, zne_preds_readout)

    return VqcNoiseSweepRunResult(
        noise_profile=task.noise_profile,
        noise_level=task.noise_level,
        shots=int(task.shots),
        seed=seed,
        ansatz=str(vc.ansatz),
        training_time=float(history.total_training_time),
        inference_time=float(inference_time),
        final_loss=float(history.best_loss),
        test_accuracy_raw=acc_raw,
        balanced_accuracy_raw=bal_raw,
        mcc_raw=mcc_raw,
        test_accuracy_readout=None if readout_metrics is None else readout_metrics[0],
        balanced_accuracy_readout=None if readout_metrics is None else readout_metrics[1],
        mcc_readout=None if readout_metrics is None else readout_metrics[2],
        test_accuracy_zne=None if zne_metrics is None else zne_metrics[0],
        balanced_accuracy_zne=None if zne_metrics is None else zne_metrics[1],
        mcc_zne=None if zne_metrics is None else zne_metrics[2],
        test_accuracy_readout_zne=None
        if readout_zne_metrics is None
        else readout_zne_metrics[0],
        balanced_accuracy_readout_zne=None
        if readout_zne_metrics is None
        else readout_zne_metrics[1],
        mcc_readout_zne=None if readout_zne_metrics is None else readout_zne_metrics[2],
    )


def _vqc_noise_worker(task: VqcNoiseSweepTask) -> VqcNoiseSweepRunResult:
    p = _vqc_noise_pool
    vc = build_circuit(**p["circuit_kwargs"])
    return _single_noise_run(
        vc=vc,
        X_train=p["X_train"],
        y_train=p["y_train"],
        X_test=p["X_test"],
        y_test=p["y_test"],
        task=task,
        max_iter=p["max_iter"],
        shot_schedule=p["shot_schedule"],
        decision_rule=p["decision_rule"],
        observable=p["observable"],
        loss_name=p["loss_name"],
        expectation_qubit=p["expectation_qubit"],
        zne_scales=p["zne_scales"],
        zne_degree=p["zne_degree"],
        single_gate_error_ratio=p["single_gate_error_ratio"],
        readout_error_rate=p["readout_error_rate"],
        backend_noise_model=p["backend_noise_model"],
        apply_readout_correction=p["apply_readout_correction"],
        apply_zne=p["apply_zne"],
        log_interval=p["log_interval"],
    )


def run_vqc_noise_sweep(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    vc_builder: Callable[[], VariationalClassifier] | None = None,
    circuit_kwargs: dict[str, Any] | None = None,
    depolarizing_rates: Sequence[float] = (0.001, 0.005, 0.01, 0.02, 0.05),
    include_backend_model: bool = False,
    backend_noise_model: Any | None = None,
    shot_budgets: Sequence[int] = (8192, 4096, 2048, 1024, 512),
    seeds: Sequence[int] = tuple(range(10)),
    max_iter: int = 200,
    shot_schedule: dict[int, int] | None = None,
    decision_rule: DecisionRule = "argmax",
    observable: MeasurementObservable = "bitstring_probs",
    loss_name: LossName = "softmax_nll",
    expectation_qubit: int = 0,
    single_gate_error_ratio: float = 0.2,
    readout_error_rate: float = 0.02,
    apply_readout_correction: bool = True,
    apply_zne: bool = True,
    zne_scales: Sequence[float] = (1.0, 2.0, 3.0),
    zne_degree: int = 1,
    mlflow_experiment: str | None = None,
    mlflow_run_prefix: str = "vqc-noise-design",
    use_cache: bool = True,
    force_rerun: bool = False,
    verbose: bool = True,
    max_workers: int | None = None,
    use_tqdm: bool = True,
    log_interval: int = 20,
) -> VqcNoiseSweepResults:
    """
    Sweep simulated VQC over noise levels, shot budgets, and seeds.

    This helper is designed for §5.6-style analysis:
      - 5-level depolarising noise sweep
      - optional backend-specific noise profile
      - readout correction and ZNE mitigation variants
      - shot-budget sensitivity (8192 → 512)
    """
    if (max_workers is not None and max_workers > 1) and circuit_kwargs is None:
        raise ValueError("circuit_kwargs is required when max_workers > 1.")

    factory = _make_vqc_factory(vc_builder, circuit_kwargs)
    probe_vc = factory()

    mlflow_available = False
    if mlflow_experiment:
        try:
            import mlflow

            _set_mlflow_tracking_uri()
            mlflow.set_experiment(mlflow_experiment)
            mlflow_available = True
        except ImportError:
            mlflow_available = False
            if verbose:
                print("Warning: MLflow not available; skipping noise-sweep logging.")

    task_grid: list[VqcNoiseSweepTask] = []
    for seed in seeds:
        for shots in shot_budgets:
            for rate in depolarizing_rates:
                task_grid.append(
                    VqcNoiseSweepTask(
                        noise_profile="depolarizing",
                        noise_level=float(rate),
                        shots=int(shots),
                        seed=int(seed),
                    )
                )
            if include_backend_model:
                task_grid.append(
                    VqcNoiseSweepTask(
                        noise_profile="backend",
                        noise_level=None,
                        shots=int(shots),
                        seed=int(seed),
                    )
                )

    cache: dict[tuple[str, float | None, int, int], VqcNoiseSweepRunResult] = {}
    if (
        use_cache
        and not force_rerun
        and mlflow_available
        and mlflow_experiment
    ):
        try:
            from mlflow.tracking import MlflowClient

            client = MlflowClient()
            exp = client.get_experiment_by_name(mlflow_experiment)
            if exp is not None:
                runs = client.search_runs(
                    experiment_ids=[exp.experiment_id],
                    order_by=["end_time DESC"],
                    max_results=20_000,
                )
                for run in runs:
                    if run.info.status != "FINISHED":
                        continue
                    p = run.data.params
                    m = run.data.metrics
                    if p.get("pipeline") != "simulated_vqc_noise":
                        continue
                    if p.get("run_prefix") != mlflow_run_prefix:
                        continue
                    if p.get("ansatz") != str(probe_vc.ansatz):
                        continue
                    try:
                        prof = p["noise_profile"]
                        level = (
                            None if p.get("noise_level") in (None, "none") else float(p["noise_level"])
                        )
                        shots = int(p["shots"])
                        seed = int(p["seed"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    key = (prof, level, shots, seed)
                    if key in cache:
                        continue
                    cache[key] = VqcNoiseSweepRunResult(
                        noise_profile=prof,
                        noise_level=level,
                        shots=shots,
                        seed=seed,
                        ansatz=str(probe_vc.ansatz),
                        training_time=float(m.get("training_time", 0.0)),
                        inference_time=float(m.get("inference_time", 0.0)),
                        final_loss=float(m.get("final_loss", 0.0)),
                        test_accuracy_raw=float(m.get("test_accuracy_raw", 0.0)),
                        balanced_accuracy_raw=float(m.get("balanced_accuracy_raw", 0.0)),
                        mcc_raw=float(m.get("mcc_raw", 0.0)),
                        test_accuracy_readout=(
                            float(m["test_accuracy_readout"])
                            if "test_accuracy_readout" in m
                            else None
                        ),
                        balanced_accuracy_readout=(
                            float(m["balanced_accuracy_readout"])
                            if "balanced_accuracy_readout" in m
                            else None
                        ),
                        mcc_readout=float(m["mcc_readout"]) if "mcc_readout" in m else None,
                        test_accuracy_zne=(
                            float(m["test_accuracy_zne"]) if "test_accuracy_zne" in m else None
                        ),
                        balanced_accuracy_zne=(
                            float(m["balanced_accuracy_zne"])
                            if "balanced_accuracy_zne" in m
                            else None
                        ),
                        mcc_zne=float(m["mcc_zne"]) if "mcc_zne" in m else None,
                        test_accuracy_readout_zne=(
                            float(m["test_accuracy_readout_zne"])
                            if "test_accuracy_readout_zne" in m
                            else None
                        ),
                        balanced_accuracy_readout_zne=(
                            float(m["balanced_accuracy_readout_zne"])
                            if "balanced_accuracy_readout_zne" in m
                            else None
                        ),
                        mcc_readout_zne=(
                            float(m["mcc_readout_zne"]) if "mcc_readout_zne" in m else None
                        ),
                    )
        except Exception:
            cache = {}

    ordered: list[VqcNoiseSweepRunResult | None] = []
    pending: list[tuple[int, VqcNoiseSweepTask]] = []
    for i, task in enumerate(task_grid):
        key = (task.noise_profile, task.noise_level, task.shots, task.seed)
        if key in cache:
            ordered.append(cache[key])
            if verbose and (max_workers is None or max_workers <= 1):
                print(
                    f"[noise {i + 1}/{len(task_grid)}] "
                    f"profile={task.noise_profile} level={task.noise_level} "
                    f"shots={task.shots} seed={task.seed} (cached)"
                )
        else:
            ordered.append(None)
            pending.append((i, task))

    computed: list[VqcNoiseSweepRunResult] = []
    if pending:
        tasks_only = [t for _, t in pending]
        if max_workers is None or max_workers <= 1:
            iterable = tasks_only
            if use_tqdm:
                from tqdm.auto import tqdm as _tqdm

                iterable = _tqdm(tasks_only, desc="VQC noise sweep", total=len(tasks_only))
            for task in iterable:
                if verbose and not use_tqdm:
                    print(
                        f"profile={task.noise_profile} level={task.noise_level} "
                        f"shots={task.shots} seed={task.seed}"
                    )
                computed.append(
                    _single_noise_run(
                        vc=factory(),
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                        task=task,
                        max_iter=max_iter,
                        shot_schedule=shot_schedule,
                        decision_rule=decision_rule,
                        observable=observable,
                        loss_name=loss_name,
                        expectation_qubit=expectation_qubit,
                        zne_scales=zne_scales,
                        zne_degree=zne_degree,
                        single_gate_error_ratio=single_gate_error_ratio,
                        readout_error_rate=readout_error_rate,
                        backend_noise_model=backend_noise_model,
                        apply_readout_correction=apply_readout_correction,
                        apply_zne=apply_zne,
                        log_interval=log_interval,
                    )
                )
        else:
            assert circuit_kwargs is not None
            cfg = {
                "circuit_kwargs": dict(circuit_kwargs),
                "X_train": X_train,
                "y_train": y_train,
                "X_test": X_test,
                "y_test": y_test,
                "max_iter": max_iter,
                "shot_schedule": shot_schedule,
                "decision_rule": decision_rule,
                "observable": observable,
                "loss_name": loss_name,
                "expectation_qubit": expectation_qubit,
                "zne_scales": tuple(float(s) for s in zne_scales),
                "zne_degree": int(zne_degree),
                "single_gate_error_ratio": float(single_gate_error_ratio),
                "readout_error_rate": float(readout_error_rate),
                "backend_noise_model": backend_noise_model,
                "apply_readout_correction": bool(apply_readout_correction),
                "apply_zne": bool(apply_zne),
                "log_interval": int(log_interval),
            }
            computed = map_parallel_or_serial(
                tasks_only,
                _vqc_noise_worker,
                max_workers=max_workers,
                use_tqdm=use_tqdm,
                tqdm_desc="VQC noise sweep",
                initializer=_vqc_noise_pool_init,
                initargs=(cfg,),
            )
        for (idx, _), result in zip(pending, computed, strict=True):
            ordered[idx] = result

    out = VqcNoiseSweepResults(results=[r for r in ordered if r is not None])

    if mlflow_available and computed:
        import mlflow

        for r in computed:
            run_name = (
                f"{mlflow_run_prefix}|{r.noise_profile}|"
                f"lvl={r.noise_level if r.noise_level is not None else 'none'}|"
                f"shots={r.shots}|seed={r.seed}"
            )
            with mlflow.start_run(run_name=run_name):
                mlflow.log_params(
                    {
                        "pipeline": "simulated_vqc_noise",
                        "run_prefix": mlflow_run_prefix,
                        "noise_profile": r.noise_profile,
                        "noise_level": "none" if r.noise_level is None else float(r.noise_level),
                        "shots": int(r.shots),
                        "seed": int(r.seed),
                        "ansatz": r.ansatz,
                        "n_qubits": probe_vc.n_qubits,
                        "n_features": probe_vc.n_features,
                        "n_trainable": probe_vc.n_trainable,
                        "max_iter": int(max_iter),
                        "decision_rule": decision_rule,
                        "observable": observable,
                        "loss_name": loss_name,
                        "expectation_qubit": int(expectation_qubit),
                    }
                )
                metrics: dict[str, float] = {
                    "training_time": float(r.training_time),
                    "inference_time": float(r.inference_time),
                    "final_loss": float(r.final_loss),
                    "test_accuracy_raw": float(r.test_accuracy_raw),
                    "balanced_accuracy_raw": float(r.balanced_accuracy_raw),
                    "mcc_raw": float(r.mcc_raw),
                }
                if r.test_accuracy_readout is not None:
                    metrics["test_accuracy_readout"] = float(r.test_accuracy_readout)
                if r.balanced_accuracy_readout is not None:
                    metrics["balanced_accuracy_readout"] = float(r.balanced_accuracy_readout)
                if r.mcc_readout is not None:
                    metrics["mcc_readout"] = float(r.mcc_readout)
                if r.test_accuracy_zne is not None:
                    metrics["test_accuracy_zne"] = float(r.test_accuracy_zne)
                if r.balanced_accuracy_zne is not None:
                    metrics["balanced_accuracy_zne"] = float(r.balanced_accuracy_zne)
                if r.mcc_zne is not None:
                    metrics["mcc_zne"] = float(r.mcc_zne)
                if r.test_accuracy_readout_zne is not None:
                    metrics["test_accuracy_readout_zne"] = float(r.test_accuracy_readout_zne)
                if r.balanced_accuracy_readout_zne is not None:
                    metrics["balanced_accuracy_readout_zne"] = float(
                        r.balanced_accuracy_readout_zne
                    )
                if r.mcc_readout_zne is not None:
                    metrics["mcc_readout_zne"] = float(r.mcc_readout_zne)
                mlflow.log_metrics(metrics)

    return out
