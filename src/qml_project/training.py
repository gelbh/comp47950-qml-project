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

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize

from qiskit.primitives import StatevectorSampler
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.primitives.containers.sampler_pub import SamplerPub

from qml_project.circuit import (
    VariationalClassifier,
    batch_loss,
    counts_to_class_probs,
    predict_batch,
)

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
    n_samples = X.shape[0]
    bound_values = vc.bind(X, theta)

    ba = BindingsArray({tuple(vc.circuit.parameters): bound_values})
    pub = SamplerPub(circuit=vc.circuit, parameter_values=ba, shots=shots)
    job = sampler.run([pub])
    result = job.result()

    class_probs = np.zeros((n_samples, vc.n_classes), dtype=np.float64)
    for i in range(n_samples):
        counts = result[0].data.meas.get_counts(i)
        class_probs[i] = counts_to_class_probs(
            counts, vc.n_qubits, vc.n_classes, class_map=vc.class_map
        )

    return class_probs


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
    verbose: bool = True,
    log_interval: int = 10,
    mlflow_experiment: str | None = None,
) -> tuple[np.ndarray, TrainingHistory]:
    """
    Train the variational classifier using COBYLA optimisation.

    Uses gradient-free COBYLA optimisation, progressive shot schedule,
    and random initialisation in :math:`[-\\pi, \\pi]`.

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
                "test_shots": test_shots,
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

        class_probs = evaluate_circuit(vc, X_train, params, shots, sampler)
        loss_val = batch_loss(class_probs, y_train)

        # Track best
        if loss_val < history.best_loss:
            history.best_loss = loss_val
            history.best_weights = params.copy()

        # Log at intervals
        if n_eval % log_interval == 0 or n_eval == 1:
            train_preds = predict_batch(class_probs)
            train_acc = float(np.mean(train_preds == y_train))

            history.losses.append(loss_val)
            history.train_accuracies.append(train_acc)
            history.eval_numbers.append(n_eval)
            history.shot_counts.append(shots)

            if X_test is not None and y_test is not None:
                test_probs = evaluate_circuit(
                    vc, X_test, params, test_shots, sampler
                )
                test_preds = predict_batch(test_probs)
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
    class_probs = evaluate_circuit(vc, X, theta, shots, sampler)
    inference_time = time.perf_counter() - t0

    preds = predict_batch(class_probs)
    accuracy = float(np.mean(preds == y))

    return {
        "accuracy": accuracy,
        "predictions": preds,
        "class_probs": class_probs,
        "inference_time": inference_time,
    }


# ---------------------------------------------------------------------------
# Multi-seed experiment
# ---------------------------------------------------------------------------


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
    verbose: bool = True,
    log_interval: int = 20,
    mlflow_experiment: str | None = None,
    mlflow_run_name: str | None = None,
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
    """
    if seeds is None:
        seeds = list(range(n_seeds))

    # MLflow parent run setup
    mlflow_parent_run = None
    if mlflow_experiment:
        try:
            import mlflow
            mlflow.set_experiment(mlflow_experiment)
            mlflow_parent_run = mlflow.start_run(run_name=mlflow_run_name)
            # Get circuit metadata from a temporary instance
            temp_vc = vc_builder()
            mlflow.log_params({
                "n_seeds": len(seeds),
                "max_iter": max_iter,
                "test_shots": test_shots,
                "n_qubits": temp_vc.n_qubits,
                "n_features": temp_vc.n_features,
                "n_classes": temp_vc.n_classes,
                "n_trainable": temp_vc.n_trainable,
            })
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
        )

        # Log seed-specific metrics to nested run
        if mlflow_parent_run:
            try:
                import mlflow
                mlflow.log_params({"seed": seed})
                mlflow.log_metrics({
                    "train_accuracy": history.train_accuracies[-1] if history.train_accuracies else 0.0,
                    "test_accuracy": eval_result["accuracy"],
                    "training_time": history.total_training_time,
                    "final_loss": history.best_loss,
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


# ---------------------------------------------------------------------------
# Noise model helpers (requires qiskit-aer)
# ---------------------------------------------------------------------------


def create_depolarizing_noise_model(
    cz_error_rate: float = 0.01,
    single_gate_error_rate: float = 0.0,
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
