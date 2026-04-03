"""Unit tests for kernel analysis helpers in baselines.py."""

from __future__ import annotations

import unittest

import numpy as np

from qml_project.baselines import (
    angle_encoding_kernel,
    centered_kernel_alignment,
    compare_kernels_for_nim,
    label_kernel_binary,
)


class TestBaselineKernels(unittest.TestCase):
    def test_angle_kernel_is_symmetric_and_unit_diagonal(self) -> None:
        X = np.array([[0.0, 0.0, 0.0], [0.5, 0.25, 0.75], [1.0, 0.0, 0.5]])
        K = angle_encoding_kernel(X, X)
        self.assertTrue(np.allclose(K, K.T))
        self.assertTrue(np.allclose(np.diag(K), np.ones(len(X))))

    def test_label_kernel_binary_maps_to_plus_minus_outer_product(self) -> None:
        y = np.array([0, 1, 1, 0])
        K = label_kernel_binary(y)
        expected_vec = np.array([-1.0, 1.0, 1.0, -1.0])
        expected = np.outer(expected_vec, expected_vec)
        self.assertTrue(np.allclose(K, expected))

    def test_centered_kernel_alignment_self_is_one(self) -> None:
        K = np.array(
            [
                [1.0, 0.2, 0.3],
                [0.2, 1.0, 0.1],
                [0.3, 0.1, 1.0],
            ]
        )
        cka = centered_kernel_alignment(K, K)
        self.assertTrue(np.isclose(cka, 1.0))

    def test_compare_kernels_for_nim_returns_expected_columns(self) -> None:
        X = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.2, 0.4, 0.6],
                [0.6, 0.4, 0.2],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        )
        y = np.array([0, 1, 1, 0], dtype=np.int32)
        df = compare_kernels_for_nim(X, y)

        self.assertEqual(set(df["kernel"]), {"angle", "rbf", "poly"})
        required = {
            "kernel",
            "cka_to_target",
            "mean_same_class",
            "mean_diff_class",
            "gap_same_minus_diff",
            "trace",
        }
        self.assertTrue(required.issubset(set(df.columns)))
        self.assertEqual(len(df), 3)


if __name__ == "__main__":
    unittest.main()
