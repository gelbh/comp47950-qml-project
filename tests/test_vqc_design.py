import numpy as np
import pytest
from qiskit.primitives import StatevectorSampler

from qml_project import (
    build_circuit,
    counts_to_z_expectation,
    evaluate_circuit_outputs,
    train_classifier,
)


def test_build_circuit_supports_two_ansatze() -> None:
    vc_basic = build_circuit(
        n_qubits=3,
        n_features=3,
        n_classes=2,
        n_layers=2,
        ansatz="basic_block",
    )
    vc_ryrz = build_circuit(
        n_qubits=3,
        n_features=3,
        n_classes=2,
        n_layers=2,
        ansatz="ry_rz",
    )

    assert vc_basic.ansatz == "basic_block"
    assert vc_ryrz.ansatz == "ry_rz"

    basic_ops = vc_basic.circuit.count_ops()
    ryrz_ops = vc_ryrz.circuit.count_ops()
    assert "rx" in basic_ops
    assert "ry" in ryrz_ops


def test_counts_to_z_expectation() -> None:
    counts = {"00": 7, "01": 3}
    # Right-most bit is qubit 0: expectation is (+1 * 7 + -1 * 3) / 10 = 0.4
    exp_z = counts_to_z_expectation(counts, qubit=0)
    assert np.isclose(exp_z, 0.4)


def test_evaluate_circuit_outputs_shapes() -> None:
    vc = build_circuit(
        n_qubits=3,
        n_features=3,
        n_classes=2,
        n_layers=2,
        ansatz="basic_block",
    )
    X = np.array([[0.0, 0.2, 0.5], [0.1, 0.3, 0.4]], dtype=np.float64)
    theta = np.zeros(vc.n_trainable, dtype=np.float64)
    sampler = StatevectorSampler(seed=123)

    outputs = evaluate_circuit_outputs(
        vc,
        X,
        theta,
        shots=64,
        sampler=sampler,
        expectation_qubit=0,
    )

    assert outputs["class_probs"].shape == (2, 2)
    assert outputs["z_expectations"].shape == (2,)


def test_expectation_loss_requires_binary_classes() -> None:
    vc = build_circuit(
        n_qubits=3,
        n_features=3,
        n_classes=3,
        n_layers=2,
        ansatz="basic_block",
    )
    X = np.array([[0.0, 0.2, 0.5]], dtype=np.float64)
    y = np.array([1], dtype=np.int64)

    with pytest.raises(ValueError, match="binary classes"):
        train_classifier(
            vc,
            X,
            y,
            max_iter=1,
            seed=0,
            verbose=False,
            loss_name="cross_entropy_expectation",
        )
