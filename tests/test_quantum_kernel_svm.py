import numpy as np

from qml_project import (
    QuantumKernelResult,
    QuantumKernelSweepResults,
    build_kernel_pipeline_comparison,
    quantum_kernel_matrix,
    run_quantum_kernel_sweep,
)
from qml_project.nim import prepare_experiment_data


def test_quantum_kernel_matrix_psd_like_diagonal() -> None:
    x = np.array(
        [
            [1, 2, 3],
            [3, 2, 1],
            [4, 1, 0],
        ],
        dtype=np.int32,
    )
    k = quantum_kernel_matrix(x, x, encoding="angle", M=7)
    assert k.shape == (3, 3)
    assert np.allclose(np.diag(k), 1.0, atol=1e-10)
    assert np.allclose(k, k.T, atol=1e-10)
    assert np.all((k >= -1e-12) & (k <= 1.0 + 1e-12))


def test_run_quantum_kernel_sweep_smoke() -> None:
    data = prepare_experiment_data(k=3, M=7, M_train=5, subset_sizes=(10,), random_state=42)
    sweep = run_quantum_kernel_sweep(
        data.split.X_train,
        data.split.y_train,
        data.split.X_test,
        data.split.y_test,
        encodings=("angle",),
        train_sizes=(10, "full"),
        seeds=(0,),
        compute_win_rate=True,
        n_games_win_rate=4,
        verbose=False,
    )
    df = sweep.to_dataframe()
    assert len(df) == 2
    assert set(df["encoding"].tolist()) == {"angle"}
    assert set(df["train_size"].tolist()) == {10, 215}
    assert df["balanced_accuracy"].between(0.0, 1.0).all()
    assert df["win_rate"].between(0.0, 1.0).all()


def test_build_kernel_pipeline_comparison_qsvm_only() -> None:
    sweep = QuantumKernelSweepResults(
        results=[
            QuantumKernelResult(
                encoding="angle",
                train_size=50,
                seed=0,
                accuracy=0.8,
                balanced_accuracy=0.7,
                mcc=0.4,
                f1=0.82,
                precision=0.85,
                recall=0.8,
                train_time_s=0.01,
                inference_time_s=0.01,
                win_rate=0.9,
            )
        ],
    )
    out = build_kernel_pipeline_comparison(sweep)
    assert not out.empty
    assert set(out["pipeline"].tolist()) == {"QSVM (Quantum Kernel)"}
