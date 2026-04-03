"""Unit tests for expressibility and trainability diagnostics."""

from __future__ import annotations

import unittest

from qml_project import (
    build_circuit,
    compare_ansatz_expressibility,
    estimate_entangling_capability,
    estimate_expressibility,
    gradient_variance_vs_depth,
)


class TestExpressibilityAnalysis(unittest.TestCase):
    def test_estimate_expressibility_returns_finite_metrics(self) -> None:
        vc = build_circuit(
            n_qubits=3,
            n_features=3,
            n_classes=2,
            n_layers=3,
            ansatz="basic_block",
        )
        metrics = estimate_expressibility(
            vc,
            n_samples=32,
            n_pairs=128,
            n_bins=25,
            seed=7,
        )
        self.assertGreaterEqual(metrics.kl_divergence_to_haar, 0.0)
        self.assertGreaterEqual(metrics.mean_fidelity, 0.0)
        self.assertLessEqual(metrics.mean_fidelity, 1.0)

    def test_estimate_entangling_capability_range(self) -> None:
        vc = build_circuit(
            n_qubits=3,
            n_features=3,
            n_classes=2,
            n_layers=3,
            ansatz="ry_rz",
        )
        metrics = estimate_entangling_capability(vc, n_samples=32, seed=5)
        self.assertGreaterEqual(metrics.meyer_wallach_mean, 0.0)
        self.assertLessEqual(metrics.meyer_wallach_mean, 1.0)

    def test_gradient_variance_vs_depth_shape_and_nonnegative(self) -> None:
        df = gradient_variance_vs_depth(
            ansatz="basic_block",
            n_qubits=3,
            n_features=3,
            n_classes=2,
            depths=[2, 3],
            n_initializations=4,
            batch_size=3,
            finite_diff_eps=1e-3,
            seed=9,
        )
        self.assertEqual(df["depth"].tolist(), [2, 3])
        self.assertTrue((df["gradient_variance_mean"] >= 0.0).all())

    def test_compare_ansatz_expressibility_has_expected_columns(self) -> None:
        df = compare_ansatz_expressibility(
            ansatze=["basic_block", "ry_rz"],
            n_qubits=3,
            n_features=3,
            n_classes=2,
            n_layers=3,
            n_samples=24,
            n_pairs=96,
            n_bins=20,
            seed=3,
        )
        required = {
            "ansatz",
            "n_qubits",
            "n_layers",
            "n_trainable",
            "kl_divergence_to_haar",
            "mean_fidelity",
            "std_fidelity",
            "meyer_wallach_mean",
            "meyer_wallach_std",
        }
        self.assertTrue(required.issubset(set(df.columns)))
        self.assertEqual(set(df["ansatz"]), {"basic_block", "ry_rz"})


if __name__ == "__main__":
    unittest.main()
