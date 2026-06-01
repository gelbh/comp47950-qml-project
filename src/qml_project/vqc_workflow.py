"""VQC tuning grid, encoding transforms, OOD tuning sweep, and robustness snapshot."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from qml_project.nim.encoding import (
    amplitude_vector,
    angle_features_matrix,
    binary_angle_features_matrix,
)
from qml_project.nim.state_utils import state_tuple_from_array
from qml_project.training.noise_sweep import run_vqc_noise_sweep
from qml_project.training.ood_sweep import run_simulated_vqc_ood_sweep

VQC_ENCODINGS: tuple[str, ...] = ("angle", "amplitude", "binary")
VQC_INCLUDE_NIM_SUM_VARIANTS: tuple[bool, ...] = (True, False)
VQC_ANSAETZE: tuple[str, ...] = ("basic_block", "ry_rz")

VQC_TRAIN_SIZES: tuple[int | str, ...] = (25, 50, 100, 150, "full")
VQC_SEEDS: tuple[int, ...] = tuple(range(10))
VQC_MAX_ITER = 200
VQC_TEST_SHOTS = 300
VQC_COMPUTE_WIN_RATE = True
VQC_N_GAMES_WIN_RATE = 200

VQC_ROBUSTNESS_NOISE_LEVELS: tuple[float, ...] = (0.0, 0.005, 0.01, 0.02)
VQC_ROBUSTNESS_SHOTS: tuple[int, ...] = (512, 1024, 2048)
VQC_ROBUSTNESS_SEEDS: tuple[int, ...] = tuple(range(5))
VQC_ROBUSTNESS_MAX_ITER = 100
VQC_ROBUSTNESS_APPLY_READOUT = True
VQC_ROBUSTNESS_APPLY_ZNE = True
VQC_ROBUSTNESS_ZNE_SCALES: tuple[float, ...] = (1.0, 2.0, 3.0)


_heap_tuple_from_row = state_tuple_from_array


def transform_states_for_vqc(
    states: np.ndarray,
    *,
    encoding: str,
    M: int = 7,
    bits_per_heap: int = 3,
    include_nim_sum: bool = True,
) -> np.ndarray:
    """Map raw heap rows to VQC angle / amplitude / binary feature matrices."""
    states = np.asarray(states, dtype=np.int32)
    if encoding == "angle":
        return angle_features_matrix(
            states,
            M=M,
            include_nim_sum=include_nim_sum,
            symmetry="none",
        )
    if encoding == "amplitude":
        rows = [
            amplitude_vector(
                _heap_tuple_from_row(s),
                M=M,
                include_nim_sum=include_nim_sum,
                symmetry="none",
            )
            for s in states
        ]
        return (np.asarray(rows, dtype=np.float64) * np.pi).astype(np.float64)
    if encoding == "binary":
        return binary_angle_features_matrix(
            states,
            bits_per_heap=bits_per_heap,
            include_nim_sum=include_nim_sum,
            symmetry="none",
        )
    raise ValueError(f"Unknown encoding: {encoding!r}")


def min_vqc_layers(n_qubits: int, n_features: int) -> int:
    """Minimum ``n_layers`` so every feature parameter can be placed.

    ``build_circuit`` alternates feature/param layers starting with a feature
    layer. Each feature layer places up to ``n_qubits`` features.
    """
    return 2 * math.ceil(n_features / n_qubits)


def vqc_encoding_profile(
    encoding: str,
    *,
    include_nim_sum: bool,
    bits_per_heap: int = 3,
) -> dict[str, Any]:
    """``n_qubits`` / ``n_features`` / depth and CZ axes for one encoding × flag."""
    b = bits_per_heap
    if encoding == "angle":
        n = 4 if include_nim_sum else 3
        return {
            "n_qubits": n,
            "n_features": n,
            "cz_strategies": ("linear", "all"),
            "depths": (2, 4),
        }
    if encoding == "amplitude":
        return {
            "n_qubits": 2,
            "n_features": 4,
            "cz_strategies": ("linear",),
            "depths": (4, 6),
        }
    if encoding == "binary":
        return {
            "n_qubits": 4 * b,
            "n_features": 4 * b,
            "cz_strategies": ("linear", "all"),
            "depths": (2, 4),
        }
    raise ValueError(f"Unknown encoding: {encoding!r}")


def build_vqc_profile_summary_rows(
    *,
    encodings: Sequence[str] = VQC_ENCODINGS,
    include_nim_sum_variants: Sequence[bool] = VQC_INCLUDE_NIM_SUM_VARIANTS,
    bits_per_heap: int = 3,
) -> list[dict[str, Any]]:
    """One row per (encoding × include_nim_sum) for §5.1-style profile tables."""
    rows: list[dict[str, Any]] = []
    for enc in encodings:
        for inc in include_nim_sum_variants:
            prof = vqc_encoding_profile(
                enc, include_nim_sum=inc, bits_per_heap=bits_per_heap
            )
            need = min_vqc_layers(prof["n_qubits"], prof["n_features"])
            assert min(prof["depths"]) >= need, (
                f"{enc} include_nim_sum={inc}: depths {prof['depths']} < min viable "
                f"{need} (n_qubits={prof['n_qubits']}, n_features={prof['n_features']})"
            )
            rows.append(
                {
                    "encoding": enc,
                    "include_nim_sum": inc,
                    "n_qubits": prof["n_qubits"],
                    "n_features": prof["n_features"],
                    "min_layers": need,
                    "depths": ",".join(str(d) for d in prof["depths"]),
                    "cz_strategies": ",".join(prof["cz_strategies"]),
                }
            )
    return rows


def build_vqc_tuning_config_grid(
    *,
    encodings: Sequence[str] = VQC_ENCODINGS,
    include_nim_sum_variants: Sequence[bool] = VQC_INCLUDE_NIM_SUM_VARIANTS,
    ansatze: Sequence[str] = VQC_ANSAETZE,
    bits_per_heap: int = 3,
) -> list[dict[str, Any]]:
    """Cartesian grid: encoding × Nim-sum × ansatz × depth × ``cz_strategy``."""
    grid: list[dict[str, Any]] = []
    for encoding in encodings:
        for include_nim_sum in include_nim_sum_variants:
            prof = vqc_encoding_profile(
                encoding, include_nim_sum=include_nim_sum, bits_per_heap=bits_per_heap
            )
            ns = "T" if include_nim_sum else "F"
            for ansatz in ansatze:
                for n_layers in prof["depths"]:
                    for cz in prof["cz_strategies"]:
                        grid.append(
                            {
                                "config_id": (
                                    f"{encoding}|{ansatz}|d={n_layers}|cz={cz}|ns={ns}"
                                ),
                                "encoding": encoding,
                                "include_nim_sum": bool(include_nim_sum),
                                "ansatz": ansatz,
                                "n_qubits": prof["n_qubits"],
                                "n_features": prof["n_features"],
                                "n_classes": 2,
                                "n_layers": n_layers,
                                "cz_strategy": cz,
                                "cz_seed": 42,
                                "loss_name": "softmax_nll",
                                "symmetry": "none",
                                "observable": "bitstring_probs",
                                "decision_rule": "argmax",
                            }
                        )
    return grid


def vqc_config_grid_preview_dataframe(
    grid: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Lightweight columns for Table 5.1-style previews in the notebook."""
    keys = (
        "config_id",
        "encoding",
        "include_nim_sum",
        "ansatz",
        "n_qubits",
        "n_features",
        "n_layers",
        "cz_strategy",
    )
    return pd.DataFrame([{k: row[k] for k in keys if k in row} for row in grid])


def circuit_kwargs_from_vqc_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Select ``build_circuit`` kwargs from a VQC tuning config mapping."""
    return {
        k: cfg[k]
        for k in (
            "n_qubits",
            "n_features",
            "n_classes",
            "ansatz",
            "n_layers",
            "cz_strategy",
            "cz_seed",
        )
        if k in cfg
    }


def run_vqc_tuning_workflow_dataframe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    vqc_config_grid: Sequence[Mapping[str, Any]],
    *,
    train_sizes: Sequence[int | str] = VQC_TRAIN_SIZES,
    seeds: Sequence[int] = VQC_SEEDS,
    max_iter: int = VQC_MAX_ITER,
    test_shots: int = VQC_TEST_SHOTS,
    expectation_qubit: int = 0,
    compute_win_rate: bool = VQC_COMPUTE_WIN_RATE,
    n_games_win_rate: int = VQC_N_GAMES_WIN_RATE,
    mlflow_experiment: str | None = None,
    use_cache: bool = True,
    run_pending: bool = False,
    verbose: bool = False,
    max_workers: int | None = None,
    tqdm_desc: str = "vqc_tuning (all configs)",
    tqdm_position: int = 0,
    bits_per_heap: int = 3,
    game_m: int = 7,
) -> pd.DataFrame:
    """Sweep every grid config through ``run_simulated_vqc_ood_sweep`` and concat."""
    enc_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def _encoded_train_test(encoding: str, include_nim_sum: bool) -> tuple[np.ndarray, np.ndarray]:
        key = f"{encoding}|{int(include_nim_sum)}"
        if key not in enc_cache:
            enc_cache[key] = (
                transform_states_for_vqc(
                    X_train,
                    encoding=encoding,
                    include_nim_sum=include_nim_sum,
                    M=game_m,
                    bits_per_heap=bits_per_heap,
                ),
                transform_states_for_vqc(
                    X_test,
                    encoding=encoding,
                    include_nim_sum=include_nim_sum,
                    M=game_m,
                    bits_per_heap=bits_per_heap,
                ),
            )
        return enc_cache[key]

    frames: list[pd.DataFrame] = []
    config_pairs = [(c["encoding"], c) for c in vqc_config_grid]
    outer = tqdm(config_pairs, desc=tqdm_desc, position=tqdm_position)
    try:
        for encoding, cfg in outer:
            cfg_id = cfg["config_id"]
            outer.set_postfix_str(f"{encoding}|{cfg_id}")
            inc = bool(cfg["include_nim_sum"])
            x_tr, x_te = _encoded_train_test(encoding, inc)

            def _feature_fn_for_policy(
                states: np.ndarray, _enc: str = encoding, _inc: bool = inc
            ) -> np.ndarray:
                return transform_states_for_vqc(
                    states,
                    encoding=_enc,
                    include_nim_sum=_inc,
                    M=game_m,
                    bits_per_heap=bits_per_heap,
                )

            sweep = run_simulated_vqc_ood_sweep(
                x_tr,
                y_train,
                x_te,
                y_test,
                circuit_kwargs=circuit_kwargs_from_vqc_config(cfg),
                train_sizes=train_sizes,
                seeds=seeds,
                max_iter=max_iter,
                test_shots=test_shots,
                decision_rule=cfg["decision_rule"],
                observable=cfg["observable"],
                loss_name=cfg["loss_name"],
                expectation_qubit=expectation_qubit,
                feature_fn_for_policy=_feature_fn_for_policy,
                compute_win_rate=compute_win_rate,
                n_games_win_rate=n_games_win_rate,
                mlflow_experiment=mlflow_experiment,
                mlflow_run_prefix=f"vqc_tuning|{encoding}|{cfg_id}",
                use_cache=use_cache,
                run_pending=run_pending,
                verbose=verbose,
                max_workers=max_workers,
            )
            df = sweep.to_dataframe()
            if df.empty:
                continue
            df = df.copy()
            df["config_id"] = cfg_id
            df["encoding"] = encoding
            df["include_nim_sum"] = inc
            df["ansatz"] = cfg["ansatz"]
            df["n_layers"] = cfg["n_layers"]
            df["cz_strategy"] = cfg["cz_strategy"]
            df["symmetry"] = cfg["symmetry"]
            df["pipeline"] = "vqc"
            df["stage"] = "tuning"
            frames.append(df)
    finally:
        outer.close()

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _vqc_grid_config_by_id(
    vqc_config_grid: Sequence[Mapping[str, Any]],
    config_id: str,
) -> Mapping[str, Any]:
    for row in vqc_config_grid:
        if row.get("config_id") == config_id:
            return row
    raise ValueError(
        f"No VQC grid entry with config_id={config_id!r}. "
        "Use the same ``vqc_config_grid`` that produced ``vqc_workflow_df``."
    )


def run_vqc_robustness_snapshot_dataframe(
    vqc_workflow_df: pd.DataFrame,
    vqc_config_grid: Sequence[Mapping[str, Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    depolarizing_rates: Sequence[float] = VQC_ROBUSTNESS_NOISE_LEVELS,
    shot_budgets: Sequence[int] = VQC_ROBUSTNESS_SHOTS,
    seeds: Sequence[int] = VQC_ROBUSTNESS_SEEDS,
    max_iter: int = VQC_ROBUSTNESS_MAX_ITER,
    apply_readout_correction: bool = VQC_ROBUSTNESS_APPLY_READOUT,
    apply_zne: bool = VQC_ROBUSTNESS_APPLY_ZNE,
    zne_scales: Sequence[float] = VQC_ROBUSTNESS_ZNE_SCALES,
    expectation_qubit: int = 0,
    mlflow_experiment: str | None = None,
    use_cache: bool = True,
    verbose: bool = False,
    max_workers: int | None = None,
    bits_per_heap: int = 3,
    game_m: int = 7,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Noise × shots sweep on the best ``config_id`` at max ``train_size`` in *tuning*.

    Returns ``(dataframe, meta)``. ``meta`` is ``None`` when *tuning* is empty or
    there are no rows at the largest train size. Otherwise ``meta`` contains at
    least ``top_config_id``, ``mean_balanced_accuracy``, and ``top_encoding``.
    """
    if vqc_workflow_df.empty or "train_size" not in vqc_workflow_df.columns:
        return pd.DataFrame(), None

    full_train_size = vqc_workflow_df["train_size"].max()
    at_full = vqc_workflow_df.loc[vqc_workflow_df["train_size"] == full_train_size]
    if at_full.empty or "balanced_accuracy" not in at_full.columns:
        return pd.DataFrame(), None

    ranked = (
        at_full.groupby("config_id")["balanced_accuracy"]
        .mean()
        .sort_values(ascending=False)
    )
    if ranked.empty:
        return pd.DataFrame(), None

    top_config_id = str(ranked.index[0])
    mean_bal = float(ranked.iloc[0])
    top_cfg = _vqc_grid_config_by_id(vqc_config_grid, top_config_id)
    top_encoding = str(top_cfg["encoding"])
    inc = bool(top_cfg.get("include_nim_sum", True))

    x_tr = transform_states_for_vqc(
        X_train, encoding=top_encoding, include_nim_sum=inc, M=game_m, bits_per_heap=bits_per_heap
    )
    x_te = transform_states_for_vqc(
        X_test, encoding=top_encoding, include_nim_sum=inc, M=game_m, bits_per_heap=bits_per_heap
    )

    noise_result = run_vqc_noise_sweep(
        x_tr,
        y_train,
        x_te,
        y_test,
        circuit_kwargs=circuit_kwargs_from_vqc_config(top_cfg),
        depolarizing_rates=depolarizing_rates,
        shot_budgets=shot_budgets,
        seeds=seeds,
        max_iter=max_iter,
        decision_rule=top_cfg["decision_rule"],
        observable=top_cfg["observable"],
        loss_name=top_cfg["loss_name"],
        expectation_qubit=expectation_qubit,
        apply_readout_correction=apply_readout_correction,
        apply_zne=apply_zne,
        zne_scales=zne_scales,
        mlflow_experiment=mlflow_experiment,
        mlflow_run_prefix=f"vqc_robustness|{top_config_id}",
        use_cache=use_cache,
        verbose=verbose,
        max_workers=max_workers,
    )
    rdf = noise_result.to_dataframe()
    meta: dict[str, Any] = {
        "top_config_id": top_config_id,
        "mean_balanced_accuracy": mean_bal,
        "top_encoding": top_encoding,
    }
    if rdf.empty:
        return pd.DataFrame(), meta

    rdf = rdf.copy()
    rdf["config_id"] = top_config_id
    rdf["encoding"] = top_encoding
    rdf["ansatz"] = top_cfg["ansatz"]
    rdf["n_layers"] = top_cfg["n_layers"]
    rdf["cz_strategy"] = top_cfg["cz_strategy"]
    rdf["symmetry"] = top_cfg["symmetry"]
    rdf["pipeline"] = "vqc"
    rdf["stage"] = "robustness"
    return rdf, meta
