"""Nim game-play policy evaluation for classical models."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.random import Generator

from qml_project.nim.game import NimMove, NimState, apply_move, legal_moves


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
    (for the opponent). Falls back to a random legal move if none.
    """

    def policy(state: NimState, rng: Generator) -> NimMove:
        moves = legal_moves(state)
        if len(moves) == 1:
            return moves[0]

        resulting_states = np.array(
            [apply_move(state, m) for m in moves],
            dtype=np.int32,
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
