"""Variational quantum classifier circuit and decoding helpers.

The package keeps each topic in its own module:

- :mod:`qml_project.circuit.cz_pairs` — ``CZStrategy``, ``AnsatzName``, and the
  CZ-pair selection helpers.
- :mod:`qml_project.circuit.builder` — :class:`VariationalClassifier` and
  :func:`build_circuit`.
- :mod:`qml_project.circuit.decoding` — bitstring → class probabilities,
  ``<Z>`` expectation, and prediction helpers.
- :mod:`qml_project.circuit.losses` — softmax NLL (single-sample and batched).

The flat public surface is unchanged from the original ``circuit.py`` module.
"""

from __future__ import annotations

from .builder import VariationalClassifier, build_circuit
from .cz_pairs import AnsatzName, CZStrategy, cz_pairs_for_layer
from .decoding import (
    bitstring_to_class_map,
    counts_to_class_probs,
    counts_to_z_expectation,
    predict_batch,
    predict_from_probs,
)
from .losses import batch_loss, softmax_nll_loss

__all__ = [
    "AnsatzName",
    "CZStrategy",
    "VariationalClassifier",
    "build_circuit",
    "cz_pairs_for_layer",
    "bitstring_to_class_map",
    "counts_to_class_probs",
    "counts_to_z_expectation",
    "predict_batch",
    "predict_from_probs",
    "batch_loss",
    "softmax_nll_loss",
]
