"""Process-parallel task execution with tqdm progress (parent process).

Workers must not call MLflow when using the default local SQLite store; log from
the parent (per completed task via :func:`map_parallel_as_completed` /
:func:`run_pending_grid_tasks`, or after each serial task).

Parallel runs use :class:`joblib.Parallel` with the ``loky`` backend (separate
worker processes), including optional per-worker ``initializer`` / ``initargs``
for sharing read-heavy arrays across tasks.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypeVar, cast

from joblib import Parallel, delayed
from tqdm.auto import tqdm

T = TypeVar("T")
R = TypeVar("R")


def _indexed_parallel_run(i: int, task: T, worker_fn: Callable[[T], R]) -> tuple[int, R]:
    """Return ``(task_index, result)`` for unordered parallel aggregation."""
    return i, worker_fn(task)


def _parallel_pool(
    n_jobs: int,
    *,
    return_as: str,
    initializer: Callable[..., None] | None,
    initargs: tuple[Any, ...],
) -> Parallel:
    kw: dict[str, Any] = {
        "n_jobs": n_jobs,
        "backend": "loky",
        "return_as": return_as,
    }
    if initializer is not None:
        kw["initializer"] = initializer
        kw["initargs"] = initargs
    return Parallel(**kw)


def map_parallel_or_serial(
    tasks: Sequence[T],
    worker_fn: Callable[[T], R],
    *,
    max_workers: int | None,
    tqdm_desc: str = "runs",
    initializer: Callable[..., None] | None = None,
    initargs: tuple[Any, ...] = (),
) -> list[R]:
    """Map *worker_fn* over *tasks* using processes if *max_workers* > 1.

    A tqdm bar is shown in the **parent** process regardless of execution mode.

    Parameters
    ----------
    tasks
        Picklable task payloads (one per run).
    worker_fn
        Top-level or otherwise picklable callable (``spawn`` must import it).
    max_workers
        ``None`` or ``<= 1`` runs sequentially in the current process.
    tqdm_desc
        Bar description.
    initializer, initargs
        Invoked once per worker process (``loky``) so workers can share read-heavy
        arrays without duplicating them in every task.

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
        return [worker_fn(t) for t in tqdm(tasks, desc=tqdm_desc, total=n)]

    parallel = _parallel_pool(
        int(max_workers),
        return_as="generator",
        initializer=initializer,
        initargs=initargs,
    )
    gen = parallel(delayed(worker_fn)(t) for t in tasks)
    return list(tqdm(gen, total=n, desc=tqdm_desc))


def map_parallel_as_completed(
    tasks: Sequence[T],
    worker_fn: Callable[[T], R],
    *,
    max_workers: int,
    tqdm_desc: str = "runs",
    initializer: Callable[..., None] | None = None,
    initargs: tuple[Any, ...] = (),
    on_each: Callable[[int, R], None] | None = None,
) -> list[R]:
    """Like :func:`map_parallel_or_serial` for ``max_workers > 1``, but completes
    tasks out-of-order and fills result slots by index (like
    :func:`concurrent.futures.as_completed`).

    Results are returned in **task order** (same as ``executor.map``). Optional
    *on_each* is invoked in the parent as each task finishes (with task index and
    result), enabling MLflow logging without waiting for the full batch.

    Workers must not call MLflow; only *on_each* in the parent should log.
    """
    n = len(tasks)
    if n == 0:
        return []

    if max_workers <= 1:
        out: list[R] = []
        for i, t in enumerate(tqdm(tasks, desc=tqdm_desc, total=n)):
            r = worker_fn(t)
            out.append(r)
            if on_each is not None:
                on_each(i, r)
        return out

    slots: list[R | None] = [None] * n
    parallel = _parallel_pool(
        int(max_workers),
        return_as="generator_unordered",
        initializer=initializer,
        initargs=initargs,
    )
    gen = parallel(
        delayed(_indexed_parallel_run)(i, tasks[i], worker_fn) for i in range(n)
    )
    for item in tqdm(gen, total=n, desc=tqdm_desc):
        i, r = item
        slots[i] = r
        if on_each is not None:
            on_each(i, r)

    if any(s is None for s in slots):
        raise RuntimeError("parallel sweep: incomplete results")
    return cast(list[R], slots)


def run_pending_grid_tasks(
    pending: Sequence[tuple[int, T]],
    ordered: list[R | None],
    *,
    run_serial: Callable[[T], R],
    parallel_worker: Callable[[T], R],
    max_workers: int | None,
    tqdm_desc: str,
    initializer: Callable[..., None] | None = None,
    initargs: tuple[Any, ...] = (),
    on_task_complete: Callable[[int, R], None] | None = None,
) -> None:
    """Execute pending sweep tasks and write results back into *ordered* by grid index.

    *pending* is ``(grid_index, task)`` pairs in **task-run order** (the index *j*
    into this sequence is what workers and :func:`map_parallel_as_completed` use).
    For each finished task, ``ordered[grid_index]`` is set to the result, then
    *on_task_complete* is called as ``(j, result)`` so the parent process can log
    MLflow (workers must not touch MLflow with the default SQLite store).

    Serial execution (``max_workers`` ``None`` or ``<= 1``) calls *run_serial*
    in-process. Parallel execution uses *parallel_worker* with the given pool
    *initializer* / *initargs*.
    """
    tasks_only = [task for _, task in pending]
    n = len(tasks_only)
    if n == 0:
        return

    def _place_and_hook(j: int, res: R) -> None:
        ordered[pending[j][0]] = res
        if on_task_complete is not None:
            on_task_complete(j, res)

    if max_workers is None or max_workers <= 1:
        for j, task in enumerate(
            tqdm(tasks_only, total=n, desc=tqdm_desc),
        ):
            _place_and_hook(j, run_serial(task))
        return

    map_parallel_as_completed(
        tasks_only,
        parallel_worker,
        max_workers=int(max_workers),
        tqdm_desc=tqdm_desc,
        initializer=initializer,
        initargs=initargs,
        on_each=_place_and_hook,
    )


__all__ = [
    "map_parallel_as_completed",
    "map_parallel_or_serial",
    "run_pending_grid_tasks",
]
