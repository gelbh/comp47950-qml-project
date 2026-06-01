"""Ensure the demo directory is on ``sys.path`` for flat ``import viz`` / ``import play_*``."""

from __future__ import annotations

import sys
from pathlib import Path

_DEMO_DIR = Path(__file__).resolve().parent


def ensure_demo_path() -> None:
    d = str(_DEMO_DIR)
    if d not in sys.path:
        sys.path.insert(0, d)
