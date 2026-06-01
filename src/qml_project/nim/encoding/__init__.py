"""Nim-specific QML input encodings and pilot go/no-go selection.

This package implements the staged encoding strategy from the project plan:

1. **Angle encoding** — ``RY(h_i * pi / M)`` per heap; optional fourth qubit with
   ``RY(nim_sum * pi / M)`` when ``include_nim_sum`` is true.
2. **Amplitude encoding** — normalised heap amplitudes with optional Nim-sum
   coordinate in the same feature vector.
3. **Binary encoding** — heap bits plus optional fixed-width Nim-sum bit
   register; per-bit-position CZs on heap qubits mirror XOR structure.

Submodules
----------
- :mod:`qml_project.nim.encoding.circuits` — primitive circuit builders and the
  shared encoding-name / symmetry types.
- :mod:`qml_project.nim.encoding.features` — VQC feature matrices stacked over
  many states.
- :mod:`qml_project.nim.encoding.pilots` — single-pipeline pilot dataclasses,
  go/no-go gating, and the binary-scope check.
- :mod:`qml_project.nim.encoding.selection` — dual-pipeline (QSVM + VQC) gating
  and top-k-if-close ranking.

The flat public surface is unchanged from the original ``nim/encoding.py``.
Encoding rationale follows the report references [18] and [20].
"""

from __future__ import annotations

from .circuits import (
    ENCODING_CANDIDATES,
    EncodingName,
    SymmetryMode,
    amplitude_vector,
    angle_parameters,
    binary_bits,
    build_amplitude_encoding_circuit,
    build_angle_encoding_circuit,
    build_binary_encoding_circuit,
    build_encoding_circuit,
)
from .features import angle_features_matrix, binary_angle_features_matrix
from .pilots import (
    BinaryScopeCriteria,
    EncodingComparison,
    EncodingDecision,
    EncodingGoNoGoCriteria,
    EncodingMetrics,
    compare_encoding_pilots,
    evaluate_binary_scope,
    evaluate_go_no_go,
    pilot_metrics_from_observation,
    select_encodings_for_sweeps,
)
from .selection import (
    DualPipelineEncodingDecision,
    DualPipelineEncodingMetrics,
    DualPipelineGateCriteria,
    evaluate_dual_pipeline_go_no_go,
    rank_dual_pipeline_encodings,
    select_dual_pipeline_encodings,
)

__all__ = [
    "ENCODING_CANDIDATES",
    "EncodingName",
    "SymmetryMode",
    "EncodingMetrics",
    "EncodingGoNoGoCriteria",
    "EncodingDecision",
    "BinaryScopeCriteria",
    "EncodingComparison",
    "DualPipelineEncodingMetrics",
    "DualPipelineGateCriteria",
    "DualPipelineEncodingDecision",
    "angle_parameters",
    "amplitude_vector",
    "binary_bits",
    "build_angle_encoding_circuit",
    "build_amplitude_encoding_circuit",
    "build_binary_encoding_circuit",
    "build_encoding_circuit",
    "angle_features_matrix",
    "binary_angle_features_matrix",
    "evaluate_go_no_go",
    "pilot_metrics_from_observation",
    "select_encodings_for_sweeps",
    "compare_encoding_pilots",
    "evaluate_binary_scope",
    "evaluate_dual_pipeline_go_no_go",
    "select_dual_pipeline_encodings",
    "rank_dual_pipeline_encodings",
]
