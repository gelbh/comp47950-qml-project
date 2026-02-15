"""
Shared code for COMP47950 Quantum Machine Learning project.
Use for data loading, metrics, circuit helpers, or other logic reused across the notebook.
"""

from qml_project.circuit import (  # noqa: F401
    VariationalClassifier,
    build_circuit,
    bitstring_to_class_map,
    counts_to_class_probs,
    softmax_nll_loss,
    batch_loss,
    predict_from_probs,
    predict_batch,
)
