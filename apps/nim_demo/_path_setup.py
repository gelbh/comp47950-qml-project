"""Ensure the demo directory and ``src/`` are on ``sys.path``.

Streamlit Cloud installs pip deps without ``-e .`` (hashed root lockfiles break
editable installs). Adding ``src/`` makes ``import qml_project`` work from a
git checkout.
"""

from __future__ import annotations

import sys
from pathlib import Path

_DEMO_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _DEMO_DIR.parents[1]
_SRC_DIR = _REPO_ROOT / "src"


def ensure_demo_path() -> None:
    for path in (_SRC_DIR, _DEMO_DIR):
        entry = str(path)
        if entry not in sys.path:
            sys.path.insert(0, entry)
