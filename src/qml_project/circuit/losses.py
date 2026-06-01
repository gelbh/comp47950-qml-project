"""Loss functions for the variational classifier."""

from __future__ import annotations

import numpy as np


def softmax_nll_loss(
    class_probs: np.ndarray,
    true_label: int,
    *,
    eps: float = 1e-10,
) -> float:
    r"""Softmax negative-log-likelihood loss.

    .. math::

        \mathcal{L}(x, y) = -\log \frac{e^{P_y}}{\sum_k e^{P_k}}

    where :math:`P_k` are the class probabilities from measurement.
    """
    exp_p = np.exp(class_probs - np.max(class_probs))
    softmax_p = exp_p / (exp_p.sum() + eps)
    return float(-np.log(softmax_p[true_label] + eps))


def batch_loss(
    class_probs_batch: np.ndarray,
    true_labels: np.ndarray,
    *,
    eps: float = 1e-10,
) -> float:
    """Mean softmax NLL loss over a batch.

    Parameters
    ----------
    class_probs_batch : ndarray, shape ``(batch_size, n_classes)``
    true_labels : ndarray, shape ``(batch_size,)``
    """
    n = len(true_labels)
    total = 0.0
    for i in range(n):
        total += softmax_nll_loss(class_probs_batch[i], int(true_labels[i]), eps=eps)
    return total / n
