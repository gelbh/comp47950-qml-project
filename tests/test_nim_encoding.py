"""Unit tests for Nim encoding utilities."""

from __future__ import annotations

import unittest

import numpy as np

from qml_project.nim import (
    BinaryScopeCriteria,
    EncodingGoNoGoCriteria,
    PilotMetrics,
    angle_parameters,
    amplitude_vector,
    binary_bits,
    build_binary_encoding_circuit,
    build_encoding_circuit,
    compare_encoding_pilots,
    evaluate_go_no_go,
    evaluate_binary_scope,
    pilot_metrics_from_observation,
)


class TestNimEncoding(unittest.TestCase):
    def test_angle_parameters_matches_formula(self) -> None:
        state = (3, 5, 2)
        theta = angle_parameters(state, M=7)
        expected = np.array(state, dtype=np.float64) * np.pi / 7.0
        self.assertTrue(np.allclose(theta, expected))

    def test_amplitude_vector_is_normalized(self) -> None:
        vec = amplitude_vector((3, 5, 2), M=7, include_nim_sum=True)
        self.assertEqual(len(vec), 4)
        self.assertTrue(np.isclose(np.linalg.norm(vec), 1.0))

    def test_binary_bits_little_endian_per_heap(self) -> None:
        bits = binary_bits((3, 5, 2), bits_per_heap=3)
        # 3 -> 011, 5 -> 101, 2 -> 010 in little-endian order per heap.
        self.assertEqual(bits.tolist(), [1, 1, 0, 1, 0, 1, 0, 1, 0])

    def test_binary_equivariant_adds_extra_cz(self) -> None:
        plain = build_binary_encoding_circuit((3, 5, 2), symmetry="none")
        equiv = build_binary_encoding_circuit((3, 5, 2), symmetry="equivariant")
        plain_cz = plain.count_ops().get("cz", 0)
        equiv_cz = equiv.count_ops().get("cz", 0)
        self.assertGreater(equiv_cz, plain_cz)

    def test_dispatch_builder_matches_named_encoder(self) -> None:
        qc = build_encoding_circuit("binary", (3, 5, 2))
        self.assertEqual(qc.num_qubits, 9)

    def test_go_no_go_rejects_on_accuracy_threshold(self) -> None:
        metrics = PilotMetrics(
            encoding="amplitude",
            n_qubits=2,
            depth=5,
            runtime_s=0.4,
            ood_balanced_accuracy=0.62,
        )
        crit = EncodingGoNoGoCriteria(min_ood_balanced_accuracy=0.70)
        decision = evaluate_go_no_go(metrics, crit)
        self.assertFalse(decision.selected)
        self.assertTrue(any("OOD balanced accuracy" in r for r in decision.reasons))

    def test_pilot_metrics_from_observation_uses_circuit_stats(self) -> None:
        metrics = pilot_metrics_from_observation(
            "angle",
            (3, 5, 2),
            runtime_s=0.12,
            ood_balanced_accuracy=0.74,
        )
        self.assertEqual(metrics.n_qubits, 3)
        self.assertGreaterEqual(metrics.depth, 1)

    def test_compare_encoding_pilots_includes_relative_deltas(self) -> None:
        pilots = [
            PilotMetrics(
                encoding="angle",
                n_qubits=3,
                depth=2,
                runtime_s=0.1,
                ood_balanced_accuracy=0.74,
                sample_efficiency_score=0.65,
            ),
            PilotMetrics(
                encoding="binary",
                n_qubits=9,
                depth=8,
                runtime_s=0.3,
                ood_balanced_accuracy=0.79,
                sample_efficiency_score=0.69,
            ),
        ]
        rows = compare_encoding_pilots(pilots)
        binary_row = next(r for r in rows if r.encoding == "binary")
        self.assertAlmostEqual(binary_row.depth_ratio_vs_angle or 0.0, 4.0)
        self.assertAlmostEqual(binary_row.runtime_ratio_vs_angle or 0.0, 3.0)
        self.assertAlmostEqual(binary_row.accuracy_delta_vs_angle or 0.0, 0.05)
        self.assertAlmostEqual(binary_row.sample_efficiency_delta_vs_angle or 0.0, 0.04)

    def test_evaluate_binary_scope_defers_when_cost_is_disproportionate(self) -> None:
        pilots = [
            PilotMetrics(
                encoding="angle",
                n_qubits=3,
                depth=2,
                runtime_s=0.1,
                ood_balanced_accuracy=0.74,
                sample_efficiency_score=0.66,
            ),
            PilotMetrics(
                encoding="amplitude",
                n_qubits=2,
                depth=1,
                runtime_s=0.2,
                ood_balanced_accuracy=0.70,
                sample_efficiency_score=0.64,
            ),
            PilotMetrics(
                encoding="binary",
                n_qubits=9,
                depth=12,
                runtime_s=0.6,
                ood_balanced_accuracy=0.75,
                sample_efficiency_score=0.66,
            ),
        ]
        decision = evaluate_binary_scope(
            pilots,
            criteria=BinaryScopeCriteria(
                max_runtime_ratio_vs_angle=3.0,
                max_depth_ratio_vs_angle=4.0,
                min_accuracy_gain_vs_angle=0.02,
                min_sample_efficiency_gain_vs_angle=0.01,
            ),
        )
        self.assertFalse(decision.selected)
        self.assertTrue(any("runtime ratio" in r or "depth ratio" in r for r in decision.reasons))

    def test_evaluate_binary_scope_selects_when_tradeoff_is_justified(self) -> None:
        pilots = [
            PilotMetrics(
                encoding="angle",
                n_qubits=3,
                depth=2,
                runtime_s=0.1,
                ood_balanced_accuracy=0.74,
                sample_efficiency_score=0.63,
            ),
            PilotMetrics(
                encoding="amplitude",
                n_qubits=2,
                depth=1,
                runtime_s=0.2,
                ood_balanced_accuracy=0.68,
                sample_efficiency_score=0.60,
            ),
            PilotMetrics(
                encoding="binary",
                n_qubits=9,
                depth=6,
                runtime_s=0.2,
                ood_balanced_accuracy=0.80,
                sample_efficiency_score=0.69,
            ),
        ]
        decision = evaluate_binary_scope(
            pilots,
            criteria=BinaryScopeCriteria(
                max_runtime_ratio_vs_angle=3.0,
                max_depth_ratio_vs_angle=4.0,
                min_accuracy_gain_vs_angle=0.02,
                min_sample_efficiency_gain_vs_angle=0.01,
            ),
        )
        self.assertTrue(decision.selected)
        self.assertTrue(any("selected" in r for r in decision.reasons))


if __name__ == "__main__":
    unittest.main()
