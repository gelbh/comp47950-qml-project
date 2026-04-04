"""Process-parallel task execution with optional tqdm progress (parent process).

Workers must not call MLflow when using the default local SQLite store; log from
the parent after collecting results.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any, TypeVar

from tqdm.auto import tqdm

T = TypeVar("T")
R = TypeVar("R")


def map_parallel_or_serial(
    tasks: Sequence[T],
    worker_fn: Callable[[T], R],
    *,
    max_workers: int | None,
    use_tqdm: bool,
    tqdm_desc: str = "runs",
    initializer: Callable[..., None] | None = None,
    initargs: tuple[Any, ...] = (),
) -> list[R]:
    """Map *worker_fn* over *tasks* using processes if *max_workers* > 1.

    Parameters
    ----------
    tasks
        Picklable task payloads (one per run).
    worker_fn
        Top-level or otherwise picklable callable (``spawn`` must import it).
    max_workers
        ``None`` or ``<= 1`` runs sequentially in the current process.
    use_tqdm
        Show a tqdm bar in the **parent** (recommended for notebooks).
    tqdm_desc
        Bar description.
    initializer, initargs
        Passed to :class:`concurrent.futures.ProcessPoolExecutor` so workers
        can share read-heavy arrays without duplicating them in every task.

    Notes
    -----
    Scripts using ``max_workers > 1`` on platforms with ``spawn`` should use
    the usual ``if __name__ == "__main__":`` guard so workers do not re-exec
    unintended code.
    """
    n = len(tasks)
    if n == 0:
        return []

    if max_workers is None or max_workers <= 1:
        if use_tqdm:
            return [worker_fn(t) for t in tqdm(tasks, desc=tqdm_desc, total=n)]
        return [worker_fn(t) for t in tasks]

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=initializer,
        initargs=initargs,
    ) as ex:
        gen = ex.map(worker_fn, tasks, chunksize=1)
        if use_tqdm:
            return list(tqdm(gen, total=n, desc=tqdm_desc))
        return list(gen)
