"""Nim game logic for normal-play Nim.

State = tuple of heap sizes; Move = (heap_index, amount_to_remove).
Supports arbitrary number of heaps (*k*) and maximum heap size (*M*).

Terminology (project conventions):
  - **heap sizes** — the tuple of non-negative integers
  - **Nim-sum** — XOR of all heap sizes
  - **normal play** — the player who takes the last stone wins
  - **winning position** — Nim-sum != 0
  - **losing position** — Nim-sum == 0
  - **optimal move** — a move that leaves Nim-sum == 0
  - **S_3 symmetry** — heap permutation symmetry (k=3)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from operator import xor
from typing import Callable

import numpy as np
from numpy.random import Generator

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

NimState = tuple[int, ...]
"""Heap sizes, e.g. ``(3, 5, 2)`` for three heaps."""

NimMove = tuple[int, int]
"""``(heap_index, amount_to_remove)``."""

Policy = Callable[[NimState, Generator], NimMove]
"""Maps a state and an RNG to a legal move."""

# ---------------------------------------------------------------------------
# Core game functions
# ---------------------------------------------------------------------------


def nim_sum(state: NimState) -> int:
    """Return the Nim-sum (XOR of all heap sizes)."""
    return reduce(xor, state, 0)


def is_terminal(state: NimState) -> bool:
    """True when all heaps are empty (game over)."""
    return all(h == 0 for h in state)


def is_winning(state: NimState) -> bool:
    """True if the current player is in a winning position (Nim-sum != 0).

    In normal play, the player to move from a winning position can always
    force a win by zeroing the Nim-sum.
    """
    return nim_sum(state) != 0


def legal_moves(state: NimState) -> list[NimMove]:
    """Return every legal move from *state*, sorted by (heap, amount).

    A legal move removes between 1 and h_i stones from heap *i* where
    h_i > 0.  Returns an empty list only for the terminal state ``(0,…,0)``.
    """
    moves: list[NimMove] = []
    for i, h in enumerate(state):
        for amount in range(1, h + 1):
            moves.append((i, amount))
    return moves


def apply_move(state: NimState, move: NimMove) -> NimState:
    """Return the state after applying *move*.

    Raises
    ------
    ValueError
        If the move is illegal (bad index or too many stones).
    """
    heap_idx, amount = move
    if heap_idx < 0 or heap_idx >= len(state):
        raise ValueError(f"Invalid heap index {heap_idx} for state {state}")
    if amount < 1 or amount > state[heap_idx]:
        raise ValueError(
            f"Cannot remove {amount} from heap {heap_idx} "
            f"(has {state[heap_idx]} stones)"
        )
    heaps = list(state)
    heaps[heap_idx] -= amount
    return tuple(heaps)


# ---------------------------------------------------------------------------
# Optimal strategy
# ---------------------------------------------------------------------------


def optimal_move(state: NimState, rng: Generator | None = None) -> NimMove:
    """Return an optimal move using the Nim-sum strategy.

    **Winning position** (Nim-sum != 0): finds a move that leaves Nim-sum == 0.
    When multiple such moves exist, one is chosen uniformly at random.

    **Losing position** (Nim-sum == 0): every move loses against optimal play,
    so a random legal move is returned.

    Parameters
    ----------
    state : NimState
        Current heap sizes.  Must not be terminal.
    rng : numpy.random.Generator or None
        Used to break ties (winning) or choose randomly (losing).
        Created with an unseeded default if *None*.
    """
    if is_terminal(state):
        raise ValueError("No moves from terminal state")

    if rng is None:
        rng = np.random.default_rng()

    ns = nim_sum(state)

    if ns == 0:
        moves = legal_moves(state)
        return moves[int(rng.integers(len(moves)))]

    winning_moves: list[NimMove] = []
    for i, h in enumerate(state):
        target = h ^ ns
        if target < h:
            winning_moves.append((i, h - target))

    return winning_moves[int(rng.integers(len(winning_moves)))]


# ---------------------------------------------------------------------------
# Built-in policies
# ---------------------------------------------------------------------------


def random_policy(state: NimState, rng: Generator) -> NimMove:
    """Choose a uniformly random legal move."""
    moves = legal_moves(state)
    return moves[int(rng.integers(len(moves)))]


def optimal_policy(state: NimState, rng: Generator) -> NimMove:
    """Play the Nim-sum optimal strategy."""
    return optimal_move(state, rng)


# ---------------------------------------------------------------------------
# Play a full game
# ---------------------------------------------------------------------------


@dataclass
class GameRecord:
    """Record of a completed Nim game.

    Attributes
    ----------
    initial_state : NimState
        Starting heap sizes.
    moves : list[NimMove]
        Sequence of moves played.
    states : list[NimState]
        Sequence of states (length = len(moves) + 1); ``states[0]`` is the
        initial state, ``states[-1]`` is the terminal ``(0, …, 0)``.
    winner : int
        0 for player A, 1 for player B.
    """

    initial_state: NimState
    moves: list[NimMove] = field(default_factory=list)
    states: list[NimState] = field(default_factory=list)
    winner: int = -1

    def __len__(self) -> int:
        return len(self.moves)


def play_game(
    policy_a: Policy,
    policy_b: Policy,
    initial_state: NimState | None = None,
    *,
    k: int = 3,
    M: int = 7,
    rng: Generator | None = None,
) -> GameRecord:
    """Play one full game of normal-play Nim between two policies.

    Parameters
    ----------
    policy_a, policy_b : Policy
        Callable ``(state, rng) -> move`` for player 0 and player 1.
    initial_state : NimState or None
        Starting heap sizes.  Defaults to ``(M, M, …, M)`` with *k* heaps.
    k : int
        Number of heaps (used only when *initial_state* is None).
    M : int
        Maximum heap size (used only when *initial_state* is None).
    rng : numpy.random.Generator or None
        Shared RNG for both policies.  Created with an unseeded default if
        *None*.

    Returns
    -------
    GameRecord
        Contains initial state, move/state history, and winner (0 or 1).
    """
    if rng is None:
        rng = np.random.default_rng()
    if initial_state is None:
        initial_state = tuple([M] * k)

    record = GameRecord(initial_state=initial_state)
    state = initial_state
    record.states.append(state)

    policies = (policy_a, policy_b)
    turn = 0

    while not is_terminal(state):
        move = policies[turn](state, rng)
        state = apply_move(state, move)
        record.moves.append(move)
        record.states.append(state)
        turn = 1 - turn

    # The player who made the last move (created the terminal state) wins.
    record.winner = 1 - turn
    return record


def play_many(
    policy_a: Policy,
    policy_b: Policy,
    n_games: int = 500,
    *,
    initial_state: NimState | None = None,
    k: int = 3,
    M: int = 7,
    seed: int = 42,
) -> dict[str, object]:
    """Play *n_games* and return aggregate statistics.

    Parameters
    ----------
    policy_a, policy_b : Policy
        Policies for player 0 (A) and player 1 (B).
    n_games : int
        Number of games to play.
    initial_state, k, M
        Passed to :func:`play_game`.
    seed : int
        Seed for the shared RNG.

    Returns
    -------
    dict
        ``wins_a``, ``wins_b``, ``win_rate_a``, ``win_rate_b``, ``games``
        (list of :class:`GameRecord`).
    """
    rng = np.random.default_rng(seed)
    games: list[GameRecord] = []
    wins_a = 0

    for _ in range(n_games):
        rec = play_game(
            policy_a, policy_b, initial_state, k=k, M=M, rng=rng
        )
        games.append(rec)
        if rec.winner == 0:
            wins_a += 1

    wins_b = n_games - wins_a
    return {
        "wins_a": wins_a,
        "wins_b": wins_b,
        "win_rate_a": wins_a / n_games,
        "win_rate_b": wins_b / n_games,
        "games": games,
    }
