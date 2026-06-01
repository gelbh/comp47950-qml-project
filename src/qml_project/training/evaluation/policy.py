"""Wrap a trained VQC as a Nim move policy and measure win rate."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from qiskit.primitives import StatevectorSampler

from qml_project.circuit import VariationalClassifier
from qml_project.nim.game import (
    NimMove,
    NimState,
    Policy,
    apply_move,
    legal_moves,
    play_many,
    random_policy,
)
from qml_project.training.types import DecisionRule

from .circuits import _predict_from_outputs, evaluate_circuit_outputs


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
