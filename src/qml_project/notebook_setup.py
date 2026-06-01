"""Small helpers so notebook cells can recover missing OOD split state (fresh kernel)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, MutableMapping

_DEFAULT_SUBSETS = (50, 100)


def _is_qml_course_repo_root(p: Path) -> bool:
    return (p / "pyproject.toml").is_file() and (p / "notebooks").is_dir()


def _repo_root_from_package() -> Path:
    """Resolve the course repo root for ``notebooks/.workflow_cache/``.

    ``Path(__file__).parents[2]`` only works when this module is loaded from a
    source checkout (``.../src/qml_project/notebook_setup.py``). If ``qml_project``
    is imported from ``site-packages/`` (non-editable install), that heuristic
    points inside the venv and device caches are never found — Section 10.4
    always re-submits.

    Resolution order:

    1. ``QML_PROJECT_ROOT`` when it points at a directory with ``pyproject.toml``
       and a ``notebooks/`` subfolder.
    2. Walk parents of this file until (1) matches.
    3. Walk parents of :func:`pathlib.Path.cwd` until (1) matches.
    4. Treat each non-empty ``sys.path`` entry (and its parent / grandparent) as
       a candidate — picks up editable installs that only add ``.../repo/src``.
    5. Fall back to ``Path(__file__).resolve().parents[2]``.
    """

    env_raw = os.environ.get("QML_PROJECT_ROOT", "").strip()
    if env_raw:
        env_p = Path(env_raw).expanduser().resolve()
        if _is_qml_course_repo_root(env_p):
            return env_p

    here = Path(__file__).resolve()
    for p in (here.parent, *here.parents):
        if _is_qml_course_repo_root(p):
            return p

    cwd = Path.cwd().resolve()
    for p in (cwd, *cwd.parents):
        if _is_qml_course_repo_root(p):
            return p

    for raw in sys.path:
        if not raw:
            continue
        try:
            base = Path(raw).resolve()
        except OSError:
            continue
        for cand in (base, base.parent, base.parent.parent):
            if _is_qml_course_repo_root(cand):
                return cand

    return here.parents[2]


def workflow_cache_path(filename: str) -> Path:
    """On-disk artifacts under ``notebooks/.workflow_cache/`` (device payloads/results)."""
    cache_dir = _repo_root_from_package() / "notebooks" / ".workflow_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / filename


def workflow_cache_search_dirs(*, max_cwd_parents: int = 24) -> tuple[Path, ...]:
    """Ordered directories to scan for **existing** workflow pickles (read path union).

    :func:`workflow_cache_path` resolves a single primary tree (and may create it).
    Jupyter often runs with :func:`Path.cwd` set to ``.../repo/notebooks`` while
    pickles live in ``.../repo/notebooks/.workflow_cache`` — not under
    ``.../repo/notebooks/notebooks/.workflow_cache``. If the resolved repo root is
    wrong (e.g. non-editable install), the primary tree can be empty even though
    the checkout has real caches; scanning these fallbacks fixes §10.4 reuse.

    Order: primary (same parent as :func:`workflow_cache_path`), then cwd-relative
    ``notebooks/.workflow_cache``, ``.workflow_cache`` beside the notebook cwd, then
    each ``<ancestor>/notebooks/.workflow_cache`` while walking parents of cwd.
    """

    seen: set[Path] = set()
    ordered: list[Path] = []

    def add(path: Path) -> None:
        try:
            resolved = path.resolve()
        except OSError:
            return
        if resolved in seen:
            return
        if not resolved.is_dir():
            return
        seen.add(resolved)
        ordered.append(resolved)

    primary = _repo_root_from_package() / "notebooks" / ".workflow_cache"
    add(primary)

    cwd = Path.cwd().resolve()
    add(cwd / "notebooks" / ".workflow_cache")
    add(cwd / ".workflow_cache")

    for i, p in enumerate((cwd, *cwd.parents)):
        if i > max_cwd_parents:
            break
        add(p / "notebooks" / ".workflow_cache")

    return tuple(ordered)


def ensure_nim_experiment_in_globals(ns: MutableMapping[str, Any]) -> None:
    """Define ``exp``, ``split``, ``dataset``, ``subsets`` if absent (Nim §2 defaults)."""
    sp = ns.get("split")
    if sp is not None and hasattr(sp, "X_train") and hasattr(sp, "y_train"):
        return

    from qml_project.nim import prepare_experiment_data

    exp = prepare_experiment_data(
        k=3,
        M=7,
        M_train=5,
        subset_sizes=_DEFAULT_SUBSETS,
        random_state=42,
    )
    ns["exp"] = exp
    ns["split"] = exp.split
    ns["dataset"] = exp.dataset
    ns["subsets"] = exp.subsets
    print(
        f"Experiment data: train {len(exp.split.X_train)}, "
        f"test {len(exp.split.X_test)}, subsets: {list(exp.subsets.keys())}"
    )
