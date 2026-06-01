"""Shared helpers for Nim state representations.

A *Nim state* is a length-``k`` integer vector ``(h_1, ..., h_k)`` recording
heap sizes. Different parts of the project consume it as a numpy array or a
tuple. :func:`state_tuple_from_array` is the single canonical converter used
as a hashable cache key, MLflow tag value, or game-engine input.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = ["state_tuple_from_array"]


def state_tuple_from_array(state: np.ndarray | Sequence[int]) -> tuple[int, ...]:
    """Return a hashable ``tuple[int, ...]`` for one Nim state.

    Accepts a 1-D numpy array, a list, a tuple, or any sequence of ints.
    Always returns Python ``int`` elements (never numpy scalars) so the tuple
    can safely be used as a dict key or compared across modules.
    """
    arr = np.asarray(state, dtype=np.int32).ravel()
    return tuple(int(v) for v in arr)
