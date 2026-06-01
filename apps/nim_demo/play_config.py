"""Constants and type aliases for the Play page (no Streamlit)."""

from __future__ import annotations

import random
from typing import Literal

from qml_project.nim.game import NimState

STARTING_PRESETS: dict[str, NimState | None] = {
    "(1, 3, 5) — classic": (1, 3, 5),
    "(2, 3, 4) — balanced": (2, 3, 4),
    "(1, 2, 3) — short demo": (1, 2, 3),
    "(3, 4, 5) — edge of training": (3, 4, 5),
    "(3, 5, 7) — OOD": (3, 5, 7),
    "Random (heaps 1..5)": None,
}

CLASSICAL_CHOICES = ["Logistic Regression", "SVM (RBF)", "Random Forest"]

CLASSICAL_SHORT: dict[str, str] = {
    "Logistic Regression": "LR",
    "SVM (RBF)": "SVM",
    "Random Forest": "RF",
}


def classical_short(full_name: str) -> str:
    return CLASSICAL_SHORT.get(full_name, full_name)


OpponentName = Literal["VQC", "QSVM", "Classical"]
Slot = Literal["human", "opp", "a", "b"]
PIPELINES: tuple[OpponentName, ...] = ("VQC", "QSVM", "Classical")
MODES = ("You vs Model", "Model vs Model")


def resolve_preset_state(preset_key: str) -> NimState:
    value = STARTING_PRESETS.get(preset_key)
    if value is not None:
        return value
    return tuple(random.randint(1, 5) for _ in range(3))
