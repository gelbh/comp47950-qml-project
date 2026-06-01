"""Demo engine: pick the next move for each pipeline and return what it saw.

The visualization panel needs the model's *confidence* per candidate move
so the probability bars visibly correspond to the chosen move. The policy
wrappers shipped in :mod:`qml_project` deliberately randomise among
equally-good moves, which is fine for evaluation but noisy for a demo.
These helpers always pick the argmax-confidence candidate and return the
raw scores and intermediate artefacts so the UI can explain the decision.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler

from qml_project.baselines.features import prepare_features
from qml_project.device_inference import QSVMDevicePayload, VQCDevicePayload
from qml_project.nim.game import (
    NimMove,
    NimState,
    apply_move,
    is_terminal,
    legal_moves,
    nim_sum,
    optimal_move,
)
from qml_project.training.evaluation import evaluate_circuit_outputs

from loaders import ClassicalBundle  # type: ignore[import-not-found]


# ---------------------------------------------------------------------------
# Explanation containers
# ---------------------------------------------------------------------------


@dataclass
class CandidateScores:
    """Per-candidate move scores from a single pipeline."""

    moves: list[NimMove]
    resulting_states: np.ndarray
    score: np.ndarray  # higher = "more confident this is a winning move for me"
    score_label: str

    @property
    def best_index(self) -> int:
        return int(np.argmax(self.score))

    @property
    def best_move(self) -> NimMove:
        return self.moves[self.best_index]


@dataclass
class VQCExplanation:
    scores: CandidateScores
    class_probs: np.ndarray  # (n_candidates, n_classes)
    features: np.ndarray  # (n_candidates, n_features)
    z_expectations: np.ndarray
    theta: np.ndarray
    circuit: QuantumCircuit
    shots: int
    n_qubits: int
    decision_rule: str


@dataclass
class QSVMExplanation:
    scores: CandidateScores
    kernel_row: np.ndarray  # (n_candidates, n_train)
    decision_values: np.ndarray
    support_mask: np.ndarray  # (n_train,) bool
    support_vectors_raw: np.ndarray  # (n_sv, k)
    dual_coef: np.ndarray  # (n_sv,)
    intercept: float
    encoding: str
    symmetry: str


@dataclass
class ClassicalExplanation:
    scores: CandidateScores
    probabilities: np.ndarray  # (n_candidates, 2) if available, else None
    features: np.ndarray  # (n_candidates, n_features)
    feature_names: list[str]
    model_name: str
    feature_set: str


@dataclass
class OptimalExplanation:
    state: NimState
    nim_sum: int
    optimal_move: NimMove
    is_winning_for_player_to_move: bool


@dataclass
class TurnExplanation:
    """All pipelines' views of the same turn, keyed by pipeline name."""

    state: NimState
    vqc: VQCExplanation | None = None
    qsvm: QSVMExplanation | None = None
    classical: ClassicalExplanation | None = None
    optimal: OptimalExplanation = field(
        default_factory=lambda: OptimalExplanation(
            state=(0, 0, 0),
            nim_sum=0,
            optimal_move=(0, 0),
            is_winning_for_player_to_move=False,
        )
    )
    #: Wall-clock milliseconds per pipeline for the last ``build_turn_explanation`` call.
    pipeline_timings_ms: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def enumerate_candidates(state: NimState) -> tuple[list[NimMove], np.ndarray]:
    """Return every legal move and the state that would result from each."""
    if is_terminal(state):
        return [], np.empty((0, len(state)), dtype=np.int32)
    moves = legal_moves(state)
    resulting = np.array([apply_move(state, m) for m in moves], dtype=np.int32)
    return moves, resulting


def _sampler_for_demo(seed: int = 0) -> Any:
    """Return a fresh StatevectorSampler pinned to ``seed`` for determinism."""
    return StatevectorSampler(seed=int(seed))


# ---------------------------------------------------------------------------
# VQC
# ---------------------------------------------------------------------------


def explain_vqc(
    state: NimState,
    payload: VQCDevicePayload,
    *,
    shots: int = 512,
    seed: int = 0,
) -> VQCExplanation:
    """Score every legal move with the trained VQC and capture artefacts."""
    moves, resulting = enumerate_candidates(state)
    if len(moves) == 0:
        raise ValueError("Cannot pick a move from a terminal state.")

    features = payload.feature_transform(resulting)
    vc = payload.build_vqc()
    outputs = evaluate_circuit_outputs(
        vc,
        features,
        payload.theta,
        int(shots),
        _sampler_for_demo(seed),
    )
    class_probs = np.asarray(outputs["class_probs"], dtype=np.float64)
    z_expectations = np.asarray(outputs["z_expectations"], dtype=np.float64)

    # Labels: 0 = losing position, 1 = winning position (project convention).
    # Resulting state is what *the opponent* faces: we want p(losing), i.e.
    # p(class=0). Higher score => more confident the opponent is stuck.
    if class_probs.shape[1] >= 1:
        score = class_probs[:, 0].copy()
    else:  # defensive: fall back to expectation-based score
        score = 0.5 * (1.0 - z_expectations)
    score_label = "p(opponent faces losing position)"

    scores = CandidateScores(
        moves=moves,
        resulting_states=resulting,
        score=score,
        score_label=score_label,
    )
    return VQCExplanation(
        scores=scores,
        class_probs=class_probs,
        features=features,
        z_expectations=z_expectations,
        theta=np.asarray(payload.theta, dtype=np.float64),
        circuit=vc.circuit,
        shots=int(shots),
        n_qubits=int(payload.n_qubits),
        decision_rule=str(payload.decision_rule),
    )


# ---------------------------------------------------------------------------
# QSVM
# ---------------------------------------------------------------------------


def explain_qsvm(
    state: NimState,
    payload: QSVMDevicePayload,
) -> QSVMExplanation:
    """Score every legal move with the trained QSVM and expose the kernel row."""
    moves, resulting = enumerate_candidates(state)
    if len(moves) == 0:
        raise ValueError("Cannot pick a move from a terminal state.")

    model_wrap = payload.model
    kernel_row = model_wrap.kernel_to_train(resulting)

    # sklearn's SVC(kernel='precomputed').decision_function returns the margin
    # for the positive class (label 1 = winning position). A *negative* margin
    # means the resulting state is predicted class 0 (losing) -- good for us.
    # Flip the sign so higher score = more confident the move is good.
    decision_values = np.asarray(
        model_wrap.model.decision_function(kernel_row), dtype=np.float64
    ).ravel()
    score = -decision_values
    score_label = "-decision_function (higher = more confident opponent loses)"

    n_train = int(model_wrap.X_train_raw.shape[0])
    support_mask = np.zeros(n_train, dtype=bool)
    support_mask[np.asarray(payload.sv_indices, dtype=np.int64)] = True

    scores = CandidateScores(
        moves=moves,
        resulting_states=resulting,
        score=score,
        score_label=score_label,
    )
    return QSVMExplanation(
        scores=scores,
        kernel_row=np.asarray(kernel_row, dtype=np.float64),
        decision_values=decision_values,
        support_mask=support_mask,
        support_vectors_raw=np.asarray(payload.support_vectors_raw, dtype=np.int32),
        dual_coef=np.asarray(payload.dual_coef, dtype=np.float64),
        intercept=float(payload.intercept),
        encoding=str(payload.encoding),
        symmetry=str(payload.symmetry),
    )


# ---------------------------------------------------------------------------
# Classical
# ---------------------------------------------------------------------------


def explain_classical(
    state: NimState,
    bundle: ClassicalBundle,
) -> ClassicalExplanation:
    """Score every legal move with the classical baseline."""
    moves, resulting = enumerate_candidates(state)
    if len(moves) == 0:
        raise ValueError("Cannot pick a move from a terminal state.")

    features = prepare_features(resulting, feature_set=bundle.feature_set, M=bundle.M)
    model = bundle.model

    probabilities: np.ndarray
    if hasattr(model, "predict_proba"):
        raw = np.asarray(model.predict_proba(features), dtype=np.float64)
        classes = list(getattr(model, "classes_", [0, 1]))
        try:
            zero_col = classes.index(0)
        except ValueError:
            zero_col = 0
        probabilities = raw
        score = raw[:, zero_col]
    elif hasattr(model, "decision_function"):
        margin = np.asarray(model.decision_function(features), dtype=np.float64).ravel()
        score = -margin
        probabilities = np.column_stack(
            [1.0 / (1.0 + np.exp(margin)), 1.0 / (1.0 + np.exp(-margin))]
        )
    else:
        preds = np.asarray(model.predict(features), dtype=np.int64)
        score = (preds == 0).astype(np.float64)
        probabilities = np.column_stack([1.0 - preds.astype(np.float64), preds.astype(np.float64)])

    score_label = "p(opponent faces losing position)"
    scores = CandidateScores(
        moves=moves,
        resulting_states=resulting,
        score=score,
        score_label=score_label,
    )
    return ClassicalExplanation(
        scores=scores,
        probabilities=probabilities,
        features=features,
        feature_names=list(bundle.feature_names),
        model_name=bundle.name,
        feature_set=bundle.feature_set,
    )


# ---------------------------------------------------------------------------
# Optimal Nim-sum reference
# ---------------------------------------------------------------------------


def explain_optimal(state: NimState, rng: np.random.Generator | None = None) -> OptimalExplanation:
    if is_terminal(state):
        return OptimalExplanation(
            state=state, nim_sum=0, optimal_move=(0, 0), is_winning_for_player_to_move=False
        )
    ns = nim_sum(state)
    opt = optimal_move(state, rng=rng)
    return OptimalExplanation(
        state=state,
        nim_sum=int(ns),
        optimal_move=opt,
        is_winning_for_player_to_move=bool(ns != 0),
    )


# ---------------------------------------------------------------------------
# Pick-move wrappers (so the UI can query a single pipeline)
# ---------------------------------------------------------------------------


def pick_move_vqc(
    state: NimState, payload: VQCDevicePayload, *, shots: int = 512, seed: int = 0
) -> tuple[NimMove, VQCExplanation]:
    exp = explain_vqc(state, payload, shots=shots, seed=seed)
    return exp.scores.best_move, exp


def pick_move_qsvm(
    state: NimState, payload: QSVMDevicePayload
) -> tuple[NimMove, QSVMExplanation]:
    exp = explain_qsvm(state, payload)
    return exp.scores.best_move, exp


def pick_move_classical(
    state: NimState, bundle: ClassicalBundle
) -> tuple[NimMove, ClassicalExplanation]:
    exp = explain_classical(state, bundle)
    return exp.scores.best_move, exp


def build_turn_explanation(
    state: NimState,
    *,
    vqc_payload: VQCDevicePayload | None,
    qsvm_payload: QSVMDevicePayload | None,
    classical_bundle: ClassicalBundle | None,
    shots: int = 512,
    seed: int = 0,
    rng: np.random.Generator | None = None,
) -> TurnExplanation:
    """Compute every pipeline's view of ``state`` in one call.

    Only pipelines with non-None inputs are populated; the rest stay None so
    the UI can gracefully hide a tab if a payload is missing.
    """
    timings: dict[str, float] = {}
    turn = TurnExplanation(state=state, optimal=explain_optimal(state, rng=rng))
    if is_terminal(state):
        turn.pipeline_timings_ms = timings
        return turn
    if vqc_payload is not None:
        t0 = time.perf_counter()
        turn.vqc = explain_vqc(state, vqc_payload, shots=shots, seed=seed)
        timings["VQC"] = (time.perf_counter() - t0) * 1000.0
    if qsvm_payload is not None:
        t0 = time.perf_counter()
        turn.qsvm = explain_qsvm(state, qsvm_payload)
        timings["QSVM"] = (time.perf_counter() - t0) * 1000.0
    if classical_bundle is not None:
        t0 = time.perf_counter()
        turn.classical = explain_classical(state, classical_bundle)
        timings["Classical"] = (time.perf_counter() - t0) * 1000.0
    turn.pipeline_timings_ms = timings
    return turn
