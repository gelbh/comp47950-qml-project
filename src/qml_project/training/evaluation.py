"""Shot schedule, circuit execution, COBYLA training, and classifier evaluation."""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize

from qiskit.primitives import StatevectorSampler
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.primitives.containers.sampler_pub import SamplerPub

from qml_project.circuit import (
    VariationalClassifier,
    batch_loss,
    predict_batch,
)
from qml_project.nim.game import (
    NimMove,
    NimState,
    Policy,
    apply_move,
    legal_moves,
    play_many,
    random_policy,
)
from qml_project.training.noise_aer import mitigate_readout_prob_vector
from qml_project.training.types import (
    DecisionRule,
    LossName,
    MeasurementObservable,
    TrainingHistory,
)

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
    qualifying = [t for t in sorted(schedule) if eval_number >= t]
    return schedule[qualifying[-1]] if qualifying else 250


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

    mlflow_run = None
    mlflow_mod: Any = None
    if mlflow_experiment:
        try:
            import mlflow as mlflow_mod
        except ImportError:
            if verbose:
                print("Warning: MLflow not available, skipping logging")
        else:
            mlflow_mod.set_experiment(mlflow_experiment)
            mlflow_run = mlflow_mod.start_run()
            mlflow_mod.log_params({
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

        if loss_val < history.best_loss:
            history.best_loss = loss_val
            history.best_weights = params.copy()

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

    if mlflow_run is not None and mlflow_mod is not None:
        try:
            mlflow_mod.log_metrics({
                "train_accuracy": history.train_accuracies[-1] if history.train_accuracies else 0.0,
                "test_accuracy": history.test_accuracies[-1] if history.test_accuracies else 0.0,
                "final_loss": history.best_loss,
                "training_time": history.total_training_time,
                "total_evals": history.total_evals,
            })
            mlflow_mod.end_run()
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
