"""Parallel vs serial sweep equivalence (small grids)."""

import numpy as np
import pandas as pd

from qml_project import (
    run_classical_sweep,
    run_quantum_kernel_sweep,
    run_simulated_vqc_ood_sweep,
)
from qml_project.nim import angle_rad_from_heaps, prepare_experiment_data


def _sort_classical(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        ["model", "feature_set", "symmetry", "train_size", "seed"],
    ).reset_index(drop=True)


def test_classical_sweep_parallel_matches_serial() -> None:
    data = prepare_experiment_data(
        k=3, M=7, M_train=5, subset_sizes=(20,), random_state=42
    )
    common = dict(
        X_train_raw=data.split.X_train,
        y_train=data.split.y_train,
        X_test_raw=data.split.X_test,
        y_test=data.split.y_test,
        model_names=("Logistic Regression",),
        feature_sets=("raw",),
        symmetry_variants=("none",),
        train_sizes=(20, "full"),
        seeds=(0, 1),
        compute_win_rate=False,
        mlflow_experiment=None,
        verbose=False,
        use_tqdm=False,
    )
    df1 = _sort_classical(
        run_classical_sweep(**common, max_workers=1).to_dataframe()
    )
    df2 = _sort_classical(
        run_classical_sweep(**common, max_workers=2).to_dataframe()
    )
    cols = [c for c in df1.columns if c not in ("train_time_s", "inference_time_s")]
    pd.testing.assert_frame_equal(df1[cols], df2[cols], check_exact=False, rtol=0.0, atol=0.0)


def _sort_qsvm(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["encoding", "train_size", "seed"]).reset_index(
        drop=True
    )


def test_qsvm_sweep_parallel_matches_serial() -> None:
    data = prepare_experiment_data(
        k=3, M=7, M_train=5, subset_sizes=(12,), random_state=1
    )
    common = dict(
        X_train_raw=data.split.X_train,
        y_train=data.split.y_train,
        X_test_raw=data.split.X_test,
        y_test=data.split.y_test,
        encodings=("angle",),
        train_sizes=(12, "full"),
        seeds=(0, 1),
        compute_win_rate=False,
        mlflow_experiment=None,
        verbose=False,
        use_tqdm=False,
    )
    df1 = _sort_qsvm(run_quantum_kernel_sweep(**common, max_workers=1).to_dataframe())
    df2 = _sort_qsvm(run_quantum_kernel_sweep(**common, max_workers=2).to_dataframe())
    cols = [c for c in df1.columns if c not in ("train_time_s", "inference_time_s")]
    pd.testing.assert_frame_equal(df1[cols], df2[cols], check_exact=False, rtol=1e-10, atol=1e-10)


def test_sim_vqc_ood_parallel_matches_serial() -> None:
    data = prepare_experiment_data(
        k=3, M=7, M_train=5, subset_sizes=(8,), random_state=2
    )
    x_train = angle_rad_from_heaps(data.split.X_train)
    x_test = angle_rad_from_heaps(data.split.X_test)
    ck = dict(
        n_qubits=3,
        n_features=3,
        n_classes=2,
        n_layers=2,
        ansatz="ry_rz",
        cz_strategy="linear",
        cz_seed=42,
    )
    common = dict(
        X_train=x_train,
        y_train=data.split.y_train,
        X_test=x_test,
        y_test=data.split.y_test,
        circuit_kwargs=ck,
        train_sizes=(8,),
        seeds=(0, 1),
        max_iter=1,
        test_shots=16,
        feature_fn_for_policy=angle_rad_from_heaps,
        compute_win_rate=False,
        mlflow_experiment=None,
        verbose=False,
        use_tqdm=False,
    )
    df1 = run_simulated_vqc_ood_sweep(**common, max_workers=1).to_dataframe()
    df2 = run_simulated_vqc_ood_sweep(**common, max_workers=2).to_dataframe()
    c1 = df1.sort_values(["train_size", "seed"]).reset_index(drop=True)
    c2 = df2.sort_values(["train_size", "seed"]).reset_index(drop=True)
    cols = [c for c in c1.columns if c not in ("training_time", "inference_time")]
    pd.testing.assert_frame_equal(
        c1[cols],
        c2[cols],
        check_exact=False,
        rtol=1e-5,
        atol=1e-5,
    )
