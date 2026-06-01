"""MLflow logger for the architecture-diagnostics pipeline."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd

from qml_project.training.experiment_namespace import standard_tags
from qml_project.training.mlflow_helpers import log_training_run, set_mlflow_tracking_uri

from .keys import (
    PIPELINE,
    TASK_EXPRESS,
    TASK_GRAD,
    ansatze_param,
    depth_ladder_param,
    encoding_specs_param,
    metric_key_grad_mean,
    metric_key_grad_std,
    metric_key_kl,
    metric_key_mw_mean,
    metric_key_mw_std,
)


def log_architecture_diagnostics_to_mlflow(
    mlflow_experiment: str,
    *,
    encoding_specs: Sequence[tuple[str, int, int]],
    ansatze: Sequence[str],
    diagnostic_base_depth: int,
    diagnostic_depth_ladder: Sequence[int],
    n_classes: int,
    cz_strategy: str,
    cz_seed: int,
    n_samples: int,
    n_pairs: int,
    n_bins: int,
    express_seed: int,
    n_initializations: int,
    batch_size: int,
    finite_diff_eps: float,
    grad_seed: int,
    diag_rows: Sequence[Mapping[str, Any]],
    grad_df: pd.DataFrame,
) -> None:
    """Write one express run per encoding and one gradient run per ``(encoding, ansatz)``."""
    try:
        import mlflow as _mlf
    except ImportError:
        return

    set_mlflow_tracking_uri()
    _mlf.set_experiment(mlflow_experiment)

    specs_sorted = sorted(encoding_specs, key=lambda t: t[0])
    specs_param = encoding_specs_param(specs_sorted)
    ansatze_t = tuple(str(a) for a in ansatze)
    ansatze_param_v = ansatze_param(ansatze_t)
    depth_param = depth_ladder_param(diagnostic_depth_ladder)

    diag_by_enc: dict[str, list[Mapping[str, Any]]] = {}
    for row in diag_rows:
        diag_by_enc.setdefault(str(row["encoding"]), []).append(row)

    for enc, n_qubits, diag_n_features in specs_sorted:
        rows = diag_by_enc.get(enc, [])
        metrics: dict[str, float] = {}
        for r in rows:
            a = str(r["ansatz"])
            metrics[metric_key_kl(a)] = float(r["kl_to_haar"])
            metrics[metric_key_mw_mean(a)] = float(r["mw_mean"])
            metrics[metric_key_mw_std(a)] = float(r["mw_std"])
        params = {
            "pipeline": PIPELINE,
            "task": TASK_EXPRESS,
            "encoding": enc,
            "encoding_specs": specs_param,
            "ansatze": ansatze_param_v,
            "n_qubits": n_qubits,
            "n_features": diag_n_features,
            "n_classes": n_classes,
            "n_layers": diagnostic_base_depth,
            "cz_strategy": cz_strategy,
            "cz_seed": cz_seed,
            "n_samples": n_samples,
            "n_pairs": n_pairs,
            "n_bins": n_bins,
            "express_seed": express_seed,
        }
        tags = standard_tags(
            pipeline="vqc",
            stage="pilot",
            extra={"architecture_diagnostics": "expressibility", "encoding": enc},
        )
        log_training_run(
            _mlf,
            run_name=f"arch-diag|express|{enc}",
            params=params,
            metrics=metrics,
            tags=tags,
        )

    if grad_df.empty:
        return

    enc_meta = {e: (nq, nf) for e, nq, nf in specs_sorted}
    for (enc, ansatz), grp in grad_df.groupby(["encoding", "ansatz"], sort=False):
        grp = grp.sort_values("depth")
        metrics = {}
        for _, r in grp.iterrows():
            d = int(r["depth"])
            metrics[metric_key_grad_mean(d)] = float(r["gradient_variance_mean"])
            metrics[metric_key_grad_std(d)] = float(r["gradient_variance_std"])
        n_qubits, n_features = enc_meta[str(enc)]
        params = {
            "pipeline": PIPELINE,
            "task": TASK_GRAD,
            "encoding": str(enc),
            "ansatz": str(ansatz),
            "encoding_specs": specs_param,
            "ansatze": ansatze_param_v,
            "n_qubits": n_qubits,
            "n_features": n_features,
            "n_classes": n_classes,
            "depth_ladder": depth_param,
            "cz_strategy": cz_strategy,
            "cz_seed": cz_seed,
            "n_initializations": n_initializations,
            "batch_size": batch_size,
            "finite_diff_eps": finite_diff_eps,
            "grad_seed": grad_seed,
        }
        tags = standard_tags(
            pipeline="vqc",
            stage="pilot",
            ansatz=str(ansatz),
            extra={"architecture_diagnostics": "gradient", "encoding": str(enc)},
        )
        log_training_run(
            _mlf,
            run_name=f"arch-diag|grad|{enc}|{ansatz}",
            params=params,
            metrics=metrics,
            tags=tags,
        )
