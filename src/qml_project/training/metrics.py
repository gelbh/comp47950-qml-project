"""Classification metrics shared across training workflows."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef


def _metrics_from_preds(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[float, float, float]:
    """Return (accuracy, balanced_accuracy, MCC)."""
    acc = float(np.mean(y_pred == y_true))
    bal = float(balanced_accuracy_score(y_true, y_pred))
    mcc_val = float(matthews_corrcoef(y_true, y_pred))
    return acc, bal, mcc_val
