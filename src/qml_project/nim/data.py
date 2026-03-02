"""Nim state enumeration, labelling, and train/val/test splitting.

Generates the complete dataset for the Nim QML project:

* **State enumeration** — all non-terminal states for arbitrary *k* and *M*.
* **Labelling** — win/loss (Option B) and optimal-move index (Option A).
* **IID regime** — 70/15/15 stratified train/val/test split with training-size
  subsets (50, 100, 200, 500, full).
* **OOD regime** — train on states with all heaps ≤ *M_train*, test on states
  with at least one heap > *M_train*.
* **Class balance** — win/loss ratio tables for each *M*.

All splits are stratified by the binary win/loss label and use explicit
``random_state`` seeds for reproducibility.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from numpy.random import Generator
from sklearn.model_selection import train_test_split

from qml_project.nim.game import (
    NimMove,
    NimState,
    is_winning,
    legal_moves,
    nim_sum,
    optimal_move,
)

# ---------------------------------------------------------------------------
# Move-index encoding (Option A)
# ---------------------------------------------------------------------------


def move_to_index(move: NimMove, *, M: int) -> int:
    """Encode ``(heap, amount)`` as a flat class index.

    Mapping: ``heap * M + (amount - 1)``, giving indices in ``[0, k*M)``.
    """
    heap, amount = move
    return heap * M + (amount - 1)


def index_to_move(idx: int, *, M: int) -> NimMove:
    """Decode a flat class index back to ``(heap, amount)``."""
    heap, rem = divmod(idx, M)
    return (heap, rem + 1)


# ---------------------------------------------------------------------------
# State enumeration
# ---------------------------------------------------------------------------


def enumerate_states(k: int = 3, M: int = 7) -> list[NimState]:
    """Return all non-terminal states for *k* heaps with max size *M*.

    States are tuples ``(h_1, …, h_k)`` with ``0 ≤ h_i ≤ M``, excluding the
    terminal state ``(0, …, 0)``.  For *k* = 3, *M* = 7 this yields 511 states.
    """
    terminal = tuple(0 for _ in range(k))
    ranges = [range(M + 1)] * k
    return [s for s in itertools.product(*ranges) if s != terminal]


# ---------------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------------


@dataclass
class NimDataset:
    """Container for enumerated Nim states with labels and metadata.

    Attributes
    ----------
    states : np.ndarray
        Shape ``(n, k)`` — heap sizes as integers.
    is_winning : np.ndarray
        Shape ``(n,)`` — 1 for winning positions, 0 for losing.
    nim_sums : np.ndarray
        Shape ``(n,)`` — Nim-sum of each state.
    optimal_move_idx : np.ndarray
        Shape ``(n,)`` — flat move index for Option A labelling.
    k : int
        Number of heaps.
    M : int
        Maximum heap size.
    n_classes_move : int
        Total number of move templates (``k * M``).
    """

    states: np.ndarray
    is_winning: np.ndarray
    nim_sums: np.ndarray
    optimal_move_idx: np.ndarray
    k: int
    M: int
    n_classes_move: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_classes_move = self.k * self.M

    def __len__(self) -> int:
        return len(self.states)

    def to_dataframe(self) -> pd.DataFrame:
        """Return a tidy DataFrame with heap columns, labels, and Nim-sum."""
        cols = {f"h{i}": self.states[:, i] for i in range(self.k)}
        cols["nim_sum"] = self.nim_sums
        cols["is_winning"] = self.is_winning
        cols["optimal_move_idx"] = self.optimal_move_idx
        return pd.DataFrame(cols)


def generate_dataset(
    k: int = 3,
    M: int = 7,
    *,
    random_state: int = 42,
) -> NimDataset:
    """Enumerate all non-terminal states and label them.

    Parameters
    ----------
    k, M : int
        Number of heaps and maximum heap size.
    random_state : int
        Seed for the RNG used to select moves for losing positions
        (Option A labelling — any legal move is acceptable).

    Returns
    -------
    NimDataset
        Fully labelled dataset ready for splitting.
    """
    raw_states = enumerate_states(k, M)
    rng = np.random.default_rng(random_state)

    states = np.array(raw_states, dtype=np.int32)
    n = len(raw_states)
    win = np.empty(n, dtype=np.int32)
    ns = np.empty(n, dtype=np.int32)
    move_idx = np.empty(n, dtype=np.int32)

    for i, s in enumerate(raw_states):
        ns[i] = nim_sum(s)
        win[i] = int(ns[i] != 0)
        mv = optimal_move(s, rng)
        move_idx[i] = move_to_index(mv, M=M)

    return NimDataset(
        states=states,
        is_winning=win,
        nim_sums=ns,
        optimal_move_idx=move_idx,
        k=k,
        M=M,
    )


# ---------------------------------------------------------------------------
# Class balance analysis
# ---------------------------------------------------------------------------


def class_balance_table(
    M_values: Sequence[int] = (3, 4, 5, 6, 7),
    k: int = 3,
) -> pd.DataFrame:
    """Compute win/loss class balance for each *M*.

    Returns a DataFrame with columns: ``M``, ``total``, ``winning``,
    ``losing``, ``pct_losing``.
    """
    rows: list[dict] = []
    for M in M_values:
        states = enumerate_states(k, M)
        n_total = len(states)
        n_losing = sum(1 for s in states if not is_winning(s))
        rows.append(
            {
                "M": M,
                "total": n_total,
                "winning": n_total - n_losing,
                "losing": n_losing,
                "pct_losing": round(100 * n_losing / n_total, 1),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# IID splitting
# ---------------------------------------------------------------------------


@dataclass
class SplitArrays:
    """Train / validation / test arrays with index tracking."""

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    idx_train: np.ndarray
    idx_val: np.ndarray
    idx_test: np.ndarray


def iid_split(
    dataset: NimDataset,
    *,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> SplitArrays:
    """70/15/15 stratified train/val/test split on a :class:`NimDataset`.

    Stratification is on ``is_winning`` to preserve the win/loss ratio in every
    partition.
    """
    n = len(dataset)
    indices = np.arange(n)
    X = dataset.states
    y = dataset.is_winning

    remaining_frac = val_size + test_size
    X_train, X_tmp, y_train, y_tmp, idx_train, idx_tmp = train_test_split(
        X,
        y,
        indices,
        test_size=remaining_frac,
        stratify=y,
        random_state=random_state,
    )

    val_of_remaining = val_size / remaining_frac
    X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
        X_tmp,
        y_tmp,
        idx_tmp,
        test_size=1.0 - val_of_remaining,
        stratify=y_tmp,
        random_state=random_state,
    )

    return SplitArrays(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
    )


# ---------------------------------------------------------------------------
# Training-size subsets
# ---------------------------------------------------------------------------


@dataclass
class TrainSubset:
    """A stratified subset of the training data."""

    X: np.ndarray
    y: np.ndarray
    indices: np.ndarray
    size: int


def training_subsets(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sizes: Sequence[int] = (50, 100, 200),
    *,
    random_state: int = 42,
) -> dict[int | str, TrainSubset]:
    """Create stratified training-size subsets for the sample-efficiency sweep.

    Parameters
    ----------
    X_train, y_train
        Full training arrays (from :func:`iid_split`).
    sizes
        Subset sizes to generate.  Sizes ≥ ``len(X_train)`` are skipped
        (the full training set is always included as key ``"full"``).
    random_state
        Seed for reproducible sub-sampling.

    Returns
    -------
    dict
        Maps each requested *size* (int) and ``"full"`` to a
        :class:`TrainSubset`.
    """
    n_full = len(X_train)
    idx_full = np.arange(n_full)
    result: dict[int | str, TrainSubset] = {}

    for sz in sizes:
        if sz >= n_full:
            continue
        _, _, _, _, idx_keep, _ = train_test_split(
            X_train,
            y_train,
            idx_full,
            train_size=sz,
            stratify=y_train,
            random_state=random_state,
        )
        result[sz] = TrainSubset(
            X=X_train[idx_keep],
            y=y_train[idx_keep],
            indices=idx_keep,
            size=sz,
        )

    result["full"] = TrainSubset(
        X=X_train,
        y=y_train,
        indices=idx_full,
        size=n_full,
    )
    return result


# ---------------------------------------------------------------------------
# OOD splitting
# ---------------------------------------------------------------------------


@dataclass
class OODSplit:
    """Out-of-distribution train/test partition.

    Training data: all non-terminal states with heaps ≤ *M_train*.
    Test data: states with at least one heap > *M_train* (and all heaps ≤ *M_test*).
    """

    train_dataset: NimDataset
    test_dataset: NimDataset
    M_train: int
    M_test: int

    @property
    def X_train(self) -> np.ndarray:
        return self.train_dataset.states

    @property
    def X_test(self) -> np.ndarray:
        return self.test_dataset.states

    @property
    def y_train(self) -> np.ndarray:
        return self.train_dataset.is_winning

    @property
    def y_test(self) -> np.ndarray:
        return self.test_dataset.is_winning


def ood_split(
    k: int = 3,
    M_train: int = 5,
    M_test: int = 7,
    *,
    random_state: int = 42,
) -> OODSplit:
    """Generate the OOD regime: train on small boards, test on larger unseen ones.

    Parameters
    ----------
    k : int
        Number of heaps.
    M_train : int
        Maximum heap size for training states.  All non-terminal states with
        every heap ≤ *M_train* form the training set.
    M_test : int
        Maximum heap size for the full state space.  States with at least one
        heap > *M_train* (and all heaps ≤ *M_test*) form the test set.
    random_state : int
        Seed for labelling RNG (move labels for losing positions).

    Returns
    -------
    OODSplit
        Contains fully labelled train and test :class:`NimDataset` objects.
    """
    train_ds = generate_dataset(k, M_train, random_state=random_state)

    all_states = enumerate_states(k, M_test)
    ood_states = [s for s in all_states if max(s) > M_train]

    rng = np.random.default_rng(random_state)
    states_arr = np.array(ood_states, dtype=np.int32)
    n = len(ood_states)
    win = np.empty(n, dtype=np.int32)
    ns_arr = np.empty(n, dtype=np.int32)
    move_idx = np.empty(n, dtype=np.int32)

    for i, s in enumerate(ood_states):
        ns_arr[i] = nim_sum(s)
        win[i] = int(ns_arr[i] != 0)
        mv = optimal_move(s, rng)
        move_idx[i] = move_to_index(mv, M=M_test)

    test_ds = NimDataset(
        states=states_arr,
        is_winning=win,
        nim_sums=ns_arr,
        optimal_move_idx=move_idx,
        k=k,
        M=M_test,
    )

    return OODSplit(
        train_dataset=train_ds,
        test_dataset=test_ds,
        M_train=M_train,
        M_test=M_test,
    )


# ---------------------------------------------------------------------------
# Feature normalisation helpers
# ---------------------------------------------------------------------------


def normalise_states(
    states: np.ndarray,
    M_max: int = 7,
) -> np.ndarray:
    """Divide heap sizes by a fixed *M_max* to get features in [0, 1].

    Uses a **global constant** (not per-split statistics) so that feature ranges
    are consistent across IID and OOD regimes.
    """
    return states.astype(np.float64) / M_max


# ---------------------------------------------------------------------------
# Convenience: prepare everything in one call
# ---------------------------------------------------------------------------


@dataclass
class NimExperimentData:
    """All data needed for the IID and OOD experimental regimes."""

    dataset: NimDataset
    iid: SplitArrays
    subsets: dict[int | str, TrainSubset]
    ood: OODSplit
    balance_table: pd.DataFrame


def prepare_experiment_data(
    k: int = 3,
    M: int = 7,
    *,
    M_train_ood: int = 5,
    subset_sizes: Sequence[int] = (50, 100, 200),
    random_state: int = 42,
) -> NimExperimentData:
    """One-call setup for the full Nim ML experiment.

    Generates the complete dataset, IID splits, training-size subsets,
    OOD split, and class balance table.

    Parameters
    ----------
    k, M : int
        Primary configuration (heaps, max size).
    M_train_ood : int
        OOD training cutoff (train on heaps ≤ this, test on larger).
    subset_sizes : sequence of int
        Training-size subsets for sample-efficiency sweep.
    random_state : int
        Master seed for all splitting and labelling.
    """
    dataset = generate_dataset(k, M, random_state=random_state)
    iid = iid_split(dataset, random_state=random_state)
    subsets = training_subsets(
        iid.X_train, iid.y_train, sizes=subset_sizes, random_state=random_state
    )
    ood = ood_split(k, M_train=M_train_ood, M_test=M, random_state=random_state)
    balance = class_balance_table(M_values=range(1, M + 1), k=k)

    return NimExperimentData(
        dataset=dataset,
        iid=iid,
        subsets=subsets,
        ood=ood,
        balance_table=balance,
    )
