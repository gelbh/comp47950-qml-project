"""Sklearn model factories for classical baselines."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from qml_project.baselines.kernels import _make_angle_kernel


def create_models(
    random_state: int = 42,
    *,
    class_weight: str | dict | None = "balanced",
    M: int = 7,
    c_svc: float = 1.0,
) -> dict[str, Any]:
    """Create classifiers with class weighting.

    Returns a dict mapping model name to an unfitted sklearn estimator.
    Includes the three standard classifiers plus a quantum-inspired
    angle-encoding kernel SVM.

    ``c_svc`` sets ``C`` for both SVM variants so classical and QSVM sweeps can
    share the same regularisation budget when comparing pipelines.
    """
    c = float(c_svc)
    return {
        "SVM (RBF)": SVC(
            kernel="rbf",
            class_weight=class_weight,
            random_state=random_state,
            C=c,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight=class_weight,
            random_state=random_state,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight=class_weight,
            random_state=random_state,
        ),
        "SVM (Angle Kernel)": SVC(
            kernel=_make_angle_kernel(M),
            class_weight=class_weight,
            random_state=random_state,
            C=c,
        ),
    }
