"""
Dataset loaders for COMP47950 QML project.

Loads Iris, Wine, and Breast Cancer Wisconsin from sklearn with train/test splits
matching Selig et al. [20] Table 1 for replication. Uses stratified splits and a
fixed random_state for reproducibility.

Datasets are loaded from sklearn on each call; no caching.
"""

from typing import Literal

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split

DatasetName = Literal["iris", "wine", "breast_cancer"]

# [20] Table 1: train/test sizes
_SPLITS: dict[DatasetName, tuple[int, int]] = {
    "iris": (90, 60),
    "wine": (108, 70),
    "breast_cancer": (449, 120),
}


def load_dataset(
    name: DatasetName,
    *,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a dataset with [20]-style train/test split.

    Returns:
        X_train, X_test, y_train, y_test
    """
    if name == "iris":
        X, y = load_iris(return_X_y=True)
    elif name == "wine":
        X, y = load_wine(return_X_y=True)
    elif name == "breast_cancer":
        X, y = load_breast_cancer(return_X_y=True)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    n_train, n_test = _SPLITS[name]
    test_size = n_test / (n_train + n_test)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    return (
        np.asarray(X_train),
        np.asarray(X_test),
        np.asarray(y_train),
        np.asarray(y_test),
    )


def load_iris_splits(*, random_state: int = 42):
    """Load Iris (4 features, 3 classes). [20]: 90 train, 60 test."""
    return load_dataset("iris", random_state=random_state)


def load_wine_splits(*, random_state: int = 42):
    """Load Wine (13 features, 3 classes). [20]: 108 train, 70 test. PCA (8 or 12 PCs) applied in preprocessing."""
    return load_dataset("wine", random_state=random_state)


def load_breast_cancer_splits(*, random_state: int = 42):
    """Load Breast Cancer Wisconsin (32 features, 2 classes). [20]: 449 train, 120 test. PCA (12 or 16 PCs) applied in preprocessing."""
    return load_dataset("breast_cancer", random_state=random_state)
