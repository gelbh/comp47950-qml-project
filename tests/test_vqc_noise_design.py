import numpy as np

from qml_project import (
    build_assignment_matrix_from_symmetric_readout_error,
    default_vqc_ansatz_hypotheses,
    mitigate_readout_prob_vector,
    run_vqc_noise_sweep,
    zne_extrapolate_to_zero,
)
from qml_project.nim import angle_rad_from_heaps, prepare_experiment_data


def test_default_vqc_ansatz_hypotheses_exposes_supported_ansatze() -> None:
    hypotheses = default_vqc_ansatz_hypotheses()
    assert set(hypotheses.keys()) == {"basic_block", "ry_rz"}
    assert "parity" in hypotheses["basic_block"].hypothesis.lower()
    assert "noise" in hypotheses["ry_rz"].expected_strength.lower()


def test_readout_mitigation_probability_vector() -> None:
    assignment = build_assignment_matrix_from_symmetric_readout_error(
        n_qubits=2,
        readout_error_rate=0.1,
    )
    true_probs = np.array([0.7, 0.2, 0.1, 0.0], dtype=np.float64)
    observed = assignment @ true_probs
    corrected = mitigate_readout_prob_vector(observed, assignment)
    assert np.allclose(np.sum(corrected), 1.0)
    assert corrected.shape == true_probs.shape
    assert np.all(corrected >= 0.0)
    assert np.linalg.norm(corrected - true_probs, ord=1) < np.linalg.norm(
        observed - true_probs, ord=1
    )


def test_zne_extrapolate_to_zero_linear_case() -> None:
    scales = (1.0, 2.0, 3.0)
    values = (0.82, 0.74, 0.66)
    extrapolated = zne_extrapolate_to_zero(scales, values, degree=1)
    assert extrapolated > values[0]


def test_run_vqc_noise_sweep_smoke() -> None:
    data = prepare_experiment_data(
        k=3, M=7, M_train=5, subset_sizes=(12,), random_state=7
    )
    x_train = angle_rad_from_heaps(data.split.X_train)
    x_test = angle_rad_from_heaps(data.split.X_test)

    sweep = run_vqc_noise_sweep(
        x_train,
        data.split.y_train,
        x_test,
        data.split.y_test,
        circuit_kwargs={
            "n_qubits": 3,
            "n_features": 3,
            "n_classes": 2,
            "n_layers": 2,
            "ansatz": "ry_rz",
            "cz_strategy": "linear",
            "cz_seed": 42,
        },
        depolarizing_rates=(0.001,),
        include_backend_model=False,
        shot_budgets=(64,),
        seeds=(0,),
        max_iter=1,
        apply_readout_correction=True,
        apply_zne=True,
        zne_scales=(1.0, 2.0, 3.0),
        use_tqdm=False,
        verbose=False,
    )
    df = sweep.to_dataframe()
    assert len(df) == 1
    assert float(df.loc[0, "balanced_accuracy_raw"]) >= 0.0
    assert float(df.loc[0, "balanced_accuracy_zne"]) >= 0.0
