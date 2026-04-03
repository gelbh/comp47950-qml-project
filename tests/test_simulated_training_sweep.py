import numpy as np
import pandas as pd

from qml_project import (
    build_circuit,
    fit_power_law_learning_curve,
    run_simulated_vqc_ood_sweep,
    sample_efficiency_stat_tests,
)
from qml_project.nim import prepare_experiment_data


def _angle_features(states: np.ndarray, m_max: int = 7) -> np.ndarray:
    return (states.astype(np.float64) / float(m_max)) * np.pi


def test_sample_efficiency_stat_tests_pairwise_outputs() -> None:
    rows: list[dict[str, float | int]] = []
    for seed in range(5):
        rows.extend(
            [
                {"seed": seed, "train_size": 50, "test_accuracy": 0.60 + 0.01 * seed},
                {"seed": seed, "train_size": 100, "test_accuracy": 0.68 + 0.01 * seed},
                {"seed": seed, "train_size": 215, "test_accuracy": 0.75 + 0.01 * seed},
            ]
        )
    df = pd.DataFrame(rows)
    stats = sample_efficiency_stat_tests(
        df,
        metric="test_accuracy",
        train_sizes=(50, 100, 215),
        alpha=0.05,
    )
    assert not stats.empty
    assert set(stats["metric"].tolist()) == {"test_accuracy"}
    assert {"size_a", "size_b", "p_value_bonferroni", "cohens_d_paired"}.issubset(
        set(stats.columns)
    )
    assert len(stats) == 3


def test_fit_power_law_learning_curve_smoke() -> None:
    fit = fit_power_law_learning_curve([50, 100, 215], [0.62, 0.70, 0.79])
    assert set(fit.keys()) == {"a", "b", "c", "r2"}
    assert np.isfinite(fit["a"])
    assert np.isfinite(fit["c"])


def test_run_simulated_vqc_ood_sweep_smoke() -> None:
    data = prepare_experiment_data(k=3, M=7, M_train=5, subset_sizes=(10,), random_state=42)
    x_train = _angle_features(data.split.X_train)
    x_test = _angle_features(data.split.X_test)
    y_train = data.split.y_train
    y_test = data.split.y_test

    def vc_builder():
        return build_circuit(
            n_qubits=3,
            n_features=3,
            n_classes=2,
            n_layers=2,
            ansatz="ry_rz",
            cz_strategy="linear",
            cz_seed=42,
        )

    sweep = run_simulated_vqc_ood_sweep(
        vc_builder,
        x_train,
        y_train,
        x_test,
        y_test,
        train_sizes=(10, "full"),
        seeds=(0,),
        max_iter=1,
        test_shots=32,
        feature_fn_for_policy=_angle_features,
        compute_win_rate=True,
        n_games_win_rate=4,
        game_k=3,
        game_M=7,
        verbose=False,
    )
    df = sweep.to_dataframe()
    assert len(df) == 2
    assert set(df["train_size"].tolist()) == {10, 215}
    assert "win_rate" in df.columns
    assert df["test_accuracy"].between(0.0, 1.0).all()
