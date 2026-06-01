"""§4.3 architecture diagnostics: expressibility, entangling capability, gradient screen."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from qml_project.circuit import build_circuit
from qml_project.expressibility import (
    compare_ansatz_expressibility,
    gradient_variance_vs_depth,
)

from .load import load_architecture_diagnostics_cache
from .log import log_architecture_diagnostics_to_mlflow

ARCH_DIAGNOSTIC_ANSATZE: tuple[str, ...] = ("basic_block", "ry_rz")
ARCH_DIAGNOSTIC_DEPTH_LADDER: tuple[int, ...] = (2, 4, 6)
ARCH_DIAGNOSTIC_BASE_DEPTH = 4


def pick_safe_diagnostic_n_features(
    *,
    max_n_features: int,
    n_qubits: int,
    n_classes: int,
    ansatz_names: Sequence[str],
    depth_values: Sequence[int],
    cz_strategy: str = "linear",
    cz_seed: int = 42,
) -> int:
    """Largest feature count for which ``build_circuit`` succeeds for all ansatz × depth."""
    upper = int(max(1, min(max_n_features, n_qubits)))
    for n_feat in range(upper, 0, -1):
        ok = True
        for ansatz in ansatz_names:
            for depth in depth_values:
                try:
                    build_circuit(
                        n_qubits=n_qubits,
                        n_features=n_feat,
                        n_classes=n_classes,
                        n_layers=int(depth),
                        cz_strategy=cz_strategy,
                        cz_seed=cz_seed,
                        ansatz=ansatz,
                    )
                except KeyError:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return n_feat
    return 1


def run_architecture_diagnostics_dataframes(
    encoding_profiles: Mapping[str, Mapping[str, Any]],
    *,
    mlflow_experiment: str,
    use_cache: bool,
    ansatze: Sequence[str] = ARCH_DIAGNOSTIC_ANSATZE,
    diagnostic_depth_ladder: Sequence[int] = ARCH_DIAGNOSTIC_DEPTH_LADDER,
    diagnostic_base_depth: int = ARCH_DIAGNOSTIC_BASE_DEPTH,
    n_classes: int = 2,
    cz_strategy: str = "linear",
    cz_seed: int = 42,
    n_samples: int = 128,
    n_pairs: int = 768,
    n_bins: int = 50,
    express_seed: int = 42,
    n_initializations: int = 12,
    batch_size: int = 6,
    finite_diff_eps: float = 1e-3,
    grad_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
    """Build expressibility + gradient-ladder tables; MLflow cache on miss.

    ``encoding_profiles`` values must include ``n_qubits`` and ``n_features``
    (as in the Section 04 notebook table). Encoding order follows dict insertion
    order.

    Returns ``(diag_df, grad_df, cache_message)``. ``cache_message`` is non-``None``
    when rows were loaded from MLflow; otherwise ``None``.
    """
    ansatze_l = list(ansatze)
    depth_ladder_l = [int(d) for d in diagnostic_depth_ladder]

    encoding_specs_list: list[tuple[str, int, int]] = []
    for encoding, prof in encoding_profiles.items():
        n_qubits_enc = int(prof["n_qubits"])
        diag_n_features = pick_safe_diagnostic_n_features(
            max_n_features=int(prof["n_features"]),
            n_qubits=n_qubits_enc,
            n_classes=n_classes,
            ansatz_names=ansatze_l,
            depth_values=depth_ladder_l,
            cz_strategy=cz_strategy,
            cz_seed=cz_seed,
        )
        encoding_specs_list.append((encoding, n_qubits_enc, diag_n_features))

    specs_t = tuple(encoding_specs_list)
    ansatze_t = tuple(str(a) for a in ansatze_l)
    depths_t = tuple(depth_ladder_l)

    cached = load_architecture_diagnostics_cache(
        mlflow_experiment,
        encoding_specs=specs_t,
        ansatze=ansatze_t,
        diagnostic_base_depth=int(diagnostic_base_depth),
        diagnostic_depth_ladder=depths_t,
        n_classes=n_classes,
        cz_strategy=cz_strategy,
        cz_seed=cz_seed,
        n_samples=n_samples,
        n_pairs=n_pairs,
        n_bins=n_bins,
        express_seed=express_seed,
        n_initializations=n_initializations,
        batch_size=batch_size,
        finite_diff_eps=finite_diff_eps,
        grad_seed=grad_seed,
        use_cache=use_cache,
    )

    if cached is not None:
        diag_rows, grad_df = cached
        msg = (
            f"  Architecture diagnostics: loaded {len(diag_rows)} expressivity rows "
            f"and {len(grad_df)} gradient rows from MLflow ({mlflow_experiment!r})."
        )
        return pd.DataFrame(diag_rows), grad_df, msg

    diag_rows: list[dict[str, Any]] = []
    grad_rows: list[pd.DataFrame] = []
    for encoding, prof in encoding_profiles.items():
        n_qubits_enc = int(prof["n_qubits"])
        diag_n_features = next(nf for enc, _, nf in encoding_specs_list if enc == encoding)

        exp_df = compare_ansatz_expressibility(
            ansatze=ansatze_l,
            n_qubits=n_qubits_enc,
            n_features=diag_n_features,
            n_classes=n_classes,
            n_layers=int(diagnostic_base_depth),
            cz_strategy=cz_strategy,
            cz_seed=cz_seed,
            n_samples=n_samples,
            n_pairs=n_pairs,
            n_bins=n_bins,
            seed=express_seed,
        )
        for _, row in exp_df.iterrows():
            diag_rows.append(
                {
                    "encoding": encoding,
                    "n_qubits": n_qubits_enc,
                    "diag_n_features": diag_n_features,
                    "ansatz": row["ansatz"],
                    "kl_to_haar": float(row["kl_divergence_to_haar"]),
                    "mw_mean": float(row["meyer_wallach_mean"]),
                    "mw_std": float(row["meyer_wallach_std"]),
                }
            )

        for ansatz in ansatze_l:
            gd = gradient_variance_vs_depth(
                ansatz=ansatz,
                n_qubits=n_qubits_enc,
                n_features=diag_n_features,
                n_classes=n_classes,
                depths=depth_ladder_l,
                cz_strategy=cz_strategy,
                cz_seed=cz_seed,
                n_initializations=n_initializations,
                batch_size=batch_size,
                finite_diff_eps=finite_diff_eps,
                seed=grad_seed,
            )
            gd = gd.copy()
            gd["encoding"] = encoding
            gd["n_qubits"] = n_qubits_enc
            gd["diag_n_features"] = diag_n_features
            grad_rows.append(gd)

    grad_df = pd.concat(grad_rows, ignore_index=True) if grad_rows else pd.DataFrame()
    log_architecture_diagnostics_to_mlflow(
        mlflow_experiment,
        encoding_specs=specs_t,
        ansatze=ansatze_t,
        diagnostic_base_depth=int(diagnostic_base_depth),
        diagnostic_depth_ladder=depths_t,
        n_classes=n_classes,
        cz_strategy=cz_strategy,
        cz_seed=cz_seed,
        n_samples=n_samples,
        n_pairs=n_pairs,
        n_bins=n_bins,
        express_seed=express_seed,
        n_initializations=n_initializations,
        batch_size=batch_size,
        finite_diff_eps=finite_diff_eps,
        grad_seed=grad_seed,
        diag_rows=diag_rows,
        grad_df=grad_df,
    )

    return pd.DataFrame(diag_rows), grad_df, None


__all__ = [
    "ARCH_DIAGNOSTIC_ANSATZE",
    "ARCH_DIAGNOSTIC_BASE_DEPTH",
    "ARCH_DIAGNOSTIC_DEPTH_LADDER",
    "pick_safe_diagnostic_n_features",
    "run_architecture_diagnostics_dataframes",
]
