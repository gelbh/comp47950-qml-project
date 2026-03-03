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


def split_class_balance(
    named_splits: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Compute class balance for an arbitrary collection of label arrays.

    Parameters
    ----------
    named_splits : dict[str, np.ndarray]
        Mapping from split name (e.g. ``"Train"``, ``"Val"``, ``"OOD Test"``)
        to a 1-D binary label array (1 = winning, 0 = losing).

    Returns
    -------
    pd.DataFrame
        Columns: ``split``, ``n``, ``winning``, ``losing``, ``pct_losing``,
        ``majority_baseline``.  ``majority_baseline`` is the accuracy of
        always predicting the majority class.
    """
    rows: list[dict] = []
    for name, y in named_splits.items():
        n = len(y)
        n_winning = int(y.sum())
        n_losing = n - n_winning
        majority = max(n_winning, n_losing)
        rows.append(
            {
                "split": name,
                "n": n,
                "winning": n_winning,
                "losing": n_losing,
                "pct_losing": round(100 * n_losing / n, 1),
                "majority_baseline": round(100 * majority / n, 1),
            }
        )
    return pd.DataFrame(rows)


def majority_baseline_accuracy(y: np.ndarray) -> float:
    """Return the accuracy (0–1) of always predicting the majority class."""
    n = len(y)
    n_majority = max(int(y.sum()), n - int(y.sum()))
    return n_majority / n


def compute_class_weights(y: np.ndarray) -> dict[int, float]:
    """Compute ``sklearn``-style ``'balanced'`` class weights.

    Weights are ``n / (n_classes * n_k)`` for each class *k*, matching
    ``sklearn.utils.class_weight.compute_class_weight(..., class_weight='balanced')``.

    Returns a dict ``{0: w_losing, 1: w_winning}`` suitable for passing to
    ``class_weight`` parameters in sklearn estimators or for scaling a custom
    loss function.
    """
    n = len(y)
    n_classes = 2
    n_0 = int((y == 0).sum())
    n_1 = int((y == 1).sum())
    return {
        0: n / (n_classes * n_0),
        1: n / (n_classes * n_1),
    }


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
# S_3 heap permutation symmetry
# ---------------------------------------------------------------------------

_S3_PERMS: list[tuple[int, ...]] = list(itertools.permutations(range(3)))
"""All 6 permutations of (0, 1, 2) — the S_3 symmetric group for k=3 heaps."""


def all_heap_permutations(
    state: np.ndarray,
) -> list[tuple[int, ...]]:
    """Return all distinct permutations of a single heap-size vector.

    Parameters
    ----------
    state : array-like, shape ``(k,)``
        A single state (heap sizes).

    Returns
    -------
    list[tuple[int, ...]]
        Unique permutations, sorted lexicographically.
    """
    native = tuple(int(x) for x in state)
    return sorted(set(itertools.permutations(native)))


def augment_s3(
    X: np.ndarray,
    y: np.ndarray,
    *,
    deduplicate: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Augment a dataset by applying all S_3 heap permutations.

    For each row in *X*, generates all permutations of the heap vector.
    Labels in *y* must be permutation-invariant (e.g. win/loss based on
    Nim-sum).  For move-index labels (Option A), use :func:`augment_s3_moves`.

    Parameters
    ----------
    X : np.ndarray, shape ``(n, k)``
        Heap-size arrays.  Only ``k = 3`` is currently supported.
    y : np.ndarray, shape ``(n,)``
        Permutation-invariant labels (e.g. ``is_winning``).
    deduplicate : bool
        If *True*, drop duplicate rows that arise from states with
        repeated heap sizes (e.g. ``(3, 3, 5)`` has only 3 unique
        permutations).

    Returns
    -------
    X_aug, y_aug : np.ndarray
        Augmented arrays.  When *deduplicate* is True, expansion factor
        is ≤ 6× (exactly 6× only when all heaps are distinct in every
        row).
    """
    k = X.shape[1]
    if k != 3:
        raise ValueError(f"augment_s3 currently supports k=3, got k={k}")

    perms = np.array(_S3_PERMS, dtype=np.intp)  # (6, 3)
    n = X.shape[0]

    # Vectorised: broadcast X[i] through all 6 permutations
    X_expanded = X[:, perms]  # (n, 6, 3)
    X_flat = X_expanded.reshape(-1, k)  # (n*6, 3)
    y_flat = np.repeat(y, len(perms))

    if deduplicate:
        _, unique_idx = np.unique(X_flat, axis=0, return_index=True)
        unique_idx.sort()
        X_flat = X_flat[unique_idx]
        y_flat = y_flat[unique_idx]

    return X_flat, y_flat


def augment_s3_moves(
    X: np.ndarray,
    y_move: np.ndarray,
    *,
    M: int,
    deduplicate: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Augment a dataset with S_3 permutations, remapping move indices.

    Unlike :func:`augment_s3`, this correctly permutes Option A move labels:
    if the original move is from heap *i*, the permuted move is from the
    heap that position *i* was mapped to.

    Parameters
    ----------
    X : np.ndarray, shape ``(n, 3)``
    y_move : np.ndarray, shape ``(n,)``
        Flat move indices (``heap * M + (amount - 1)``).
    M : int
        Maximum heap size (for move encoding).
    deduplicate : bool
        If *True*, keeps only the first occurrence of each unique state
        (and its corresponding permuted move label).

    Returns
    -------
    X_aug, y_move_aug : np.ndarray
    """
    k = X.shape[1]
    if k != 3:
        raise ValueError(f"augment_s3_moves currently supports k=3, got k={k}")

    perms = np.array(_S3_PERMS, dtype=np.intp)  # (6, 3)
    inv_perms = np.empty_like(perms)
    for i, p in enumerate(perms):
        for j, v in enumerate(p):
            inv_perms[i, v] = j

    n = X.shape[0]
    X_expanded = X[:, perms]  # (n, 6, 3)
    X_flat = X_expanded.reshape(-1, k)

    # Remap move labels: original (heap, amount) → (inv_perm[heap], amount)
    heaps_orig = y_move // M        # (n,)
    amounts_m1 = y_move % M         # amount - 1

    # For each of the 6 perms, compute the new heap index
    # inv_perms[:, heap_orig] gives the new heap index for each perm
    new_heaps = inv_perms[:, heaps_orig]  # (6, n)
    new_heaps = new_heaps.T.ravel()       # (n*6,)
    amounts_flat = np.tile(amounts_m1, len(perms))
    y_flat = new_heaps * M + amounts_flat

    if deduplicate:
        _, unique_idx = np.unique(X_flat, axis=0, return_index=True)
        unique_idx.sort()
        X_flat = X_flat[unique_idx]
        y_flat = y_flat[unique_idx]

    return X_flat, y_flat


def canonical_order(
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sort heap sizes in ascending order (canonical form under S_3).

    Collapsing all permutations of a state into one canonical representative
    reduces the effective state space.  For ``k = 3, M = 7`` the 511
    non-terminal states collapse to the number of unique sorted tuples.

    Parameters
    ----------
    X : np.ndarray, shape ``(n, k)``

    Returns
    -------
    X_sorted : np.ndarray, shape ``(n, k)``
        Heap sizes sorted ascending within each row
        (``h_1 \\leq h_2 \\leq h_3``).
    sort_perms : np.ndarray, shape ``(n, k)``
        The ``argsort`` permutation applied to each row.  Useful for
        mapping a predicted move index back to the original heap ordering.
    """
    sort_perms = np.argsort(X, axis=1).astype(np.intp)
    X_sorted = np.take_along_axis(X, sort_perms, axis=1)
    return X_sorted, sort_perms


def remap_move_to_original(
    move_idx: int | np.ndarray,
    sort_perm: np.ndarray,
    *,
    M: int,
) -> int | np.ndarray:
    """Map a predicted move index from canonical heap order back to original.

    After canonical ordering, heap indices refer to sorted positions.  This
    function translates the predicted heap index back to the original
    (unsorted) heap index using the stored ``argsort`` permutation.

    Parameters
    ----------
    move_idx : int or np.ndarray
        Flat move index in the canonical (sorted) frame.
    sort_perm : np.ndarray, shape ``(k,)`` or ``(n, k)``
        The ``argsort`` permutation from :func:`canonical_order`.
    M : int
        Maximum heap size.
    """
    scalar = np.isscalar(move_idx)
    move_idx = np.atleast_1d(np.asarray(move_idx))
    sort_perm = np.atleast_2d(sort_perm)

    sorted_heap = move_idx // M
    amount_m1 = move_idx % M

    # sort_perm[i] maps original positions → sorted positions;
    # sort_perm[i, j] = original position that ended up in sorted position j.
    # So original_heap = sort_perm[i, sorted_heap[i]].
    if sort_perm.shape[0] == 1:
        original_heap = sort_perm[0, sorted_heap]
    else:
        original_heap = sort_perm[np.arange(len(move_idx)), sorted_heap]

    result = original_heap * M + amount_m1
    return int(result[0]) if scalar else result


def count_canonical_states(k: int = 3, M: int = 7) -> int:
    """Count unique states under canonical (sorted ascending) ordering.

    This is the number of non-terminal multisets
    ``{h_1, h_2, …, h_k}`` with ``0 ≤ h_i ≤ M``, excluding all-zeros.
    Equivalently, the number of *k*-element multisets from ``{0, …, M}``
    minus 1 (for the terminal state).
    """
    states = enumerate_states(k, M)
    canonical = {tuple(sorted(s)) for s in states}
    return len(canonical)


def augmentation_stats(
    X: np.ndarray,
) -> pd.DataFrame:
    """Compute per-state augmentation expansion statistics.

    For each row in *X*, reports how many unique S_3 permutations exist
    (depends on heap repetitions).

    Returns a DataFrame with columns: ``state``, ``n_unique_perms``,
    ``n_total_perms``, ``expansion_factor``.
    """
    rows: list[dict] = []
    seen: set[tuple[int, ...]] = set()
    for state in X:
        key = tuple(state)
        if key in seen:
            continue
        seen.add(key)
        n_unique = len(set(itertools.permutations(state)))
        rows.append(
            {
                "state": key,
                "n_unique_perms": n_unique,
                "n_total_perms": 6,
                "expansion_factor": n_unique,
            }
        )
    df = pd.DataFrame(rows)
    return df


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
