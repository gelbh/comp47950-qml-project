"""MLflow cache loader for the architecture-diagnostics pipeline."""

from __future__ import annotations

import math
from typing import Any, Sequence

import pandas as pd

from qml_project.training.mlflow_helpers import set_mlflow_tracking_uri
from qml_project.training.mlflow_schema import ParamKey

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


def load_architecture_diagnostics_cache(
    experiment_name: str,
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
    use_cache: bool,
) -> tuple[list[dict[str, Any]], pd.DataFrame] | None:
    """Return cached ``diag_rows`` + ``grad_df`` if every piece matches, else ``None``."""
    if not use_cache or not experiment_name:
        return None
    try:
        from mlflow.tracking import MlflowClient
    except ImportError:
        return None

    set_mlflow_tracking_uri()
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        return None

    specs_sorted = sorted(encoding_specs, key=lambda t: t[0])
    enc_meta = {e: (int(nq), int(nf)) for e, nq, nf in specs_sorted}
    encodings = [e for e, _, _ in specs_sorted]
    ansatze_t = tuple(str(a) for a in ansatze)
    depths_t = tuple(int(d) for d in diagnostic_depth_ladder)
    specs_param = encoding_specs_param(specs_sorted)
    ansatze_param_v = ansatze_param(ansatze_t)
    depth_param = depth_ladder_param(depths_t)

    want_express = set(encodings)
    got_express: dict[str, dict[str, dict[str, float]]] = {}
    want_grad = {(e, a) for e in encodings for a in ansatze_t}
    got_grad: dict[tuple[str, str], dict[int, dict[str, float]]] = {}

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["end_time DESC"],
        max_results=5000,
    )
    for run in runs:
        if run.info.status != "FINISHED":
            continue
        p = run.data.params
        m = run.data.metrics
        if p.get(ParamKey.PIPELINE) != PIPELINE:
            continue
        task = p.get(ParamKey.TASK)
        if task == TASK_EXPRESS:
            if (
                p.get("encoding_specs") != specs_param
                or p.get("ansatze") != ansatze_param_v
                or int(p.get("n_layers", -1)) != int(diagnostic_base_depth)
                or int(p.get("n_classes", -1)) != int(n_classes)
                or p.get("cz_strategy") != str(cz_strategy)
                or int(p.get("cz_seed", -1)) != int(cz_seed)
                or int(p.get("n_samples", -1)) != int(n_samples)
                or int(p.get("n_pairs", -1)) != int(n_pairs)
                or int(p.get("n_bins", -1)) != int(n_bins)
                or int(p.get("express_seed", -1)) != int(express_seed)
            ):
                continue
            enc = p.get("encoding")
            if enc is None or enc not in want_express or enc in got_express:
                continue
            exp_nq, exp_nf = enc_meta[str(enc)]
            if int(p.get("n_qubits", -1)) != exp_nq or int(p.get("n_features", -1)) != exp_nf:
                continue
            per_a: dict[str, dict[str, float]] = {}
            ok = True
            for a in ansatze_t:
                k_kl = metric_key_kl(a)
                k_m = metric_key_mw_mean(a)
                k_s = metric_key_mw_std(a)
                if k_kl not in m or k_m not in m or k_s not in m:
                    ok = False
                    break
                per_a[a] = {
                    "kl_to_haar": float(m[k_kl]),
                    "mw_mean": float(m[k_m]),
                    "mw_std": float(m[k_s]),
                }
            if ok:
                got_express[str(enc)] = per_a

        elif task == TASK_GRAD:
            if (
                p.get("encoding_specs") != specs_param
                or p.get("ansatze") != ansatze_param_v
                or p.get("depth_ladder") != depth_param
                or int(p.get("n_classes", -1)) != int(n_classes)
                or p.get("cz_strategy") != str(cz_strategy)
                or int(p.get("cz_seed", -1)) != int(cz_seed)
                or int(p.get("n_initializations", -1)) != int(n_initializations)
                or int(p.get("batch_size", -1)) != int(batch_size)
                or not math.isclose(
                    float(p.get("finite_diff_eps", -1.0)),
                    float(finite_diff_eps),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or int(p.get("grad_seed", -1)) != int(grad_seed)
            ):
                continue
            enc = p.get("encoding")
            a = p.get("ansatz")
            if enc is None or a is None:
                continue
            key = (str(enc), str(a))
            if key not in want_grad or key in got_grad:
                continue
            exp_nq, exp_nf = enc_meta[str(enc)]
            if int(p.get("n_qubits", -1)) != exp_nq or int(p.get("n_features", -1)) != exp_nf:
                continue
            by_depth: dict[int, dict[str, float]] = {}
            ok = True
            for d in depths_t:
                km = metric_key_grad_mean(d)
                ks = metric_key_grad_std(d)
                if km not in m or ks not in m:
                    ok = False
                    break
                by_depth[int(d)] = {
                    "gradient_variance_mean": float(m[km]),
                    "gradient_variance_std": float(m[ks]),
                }
            if ok:
                got_grad[key] = by_depth

    if set(got_express.keys()) != want_express or set(got_grad.keys()) != want_grad:
        return None

    diag_rows: list[dict[str, Any]] = []
    for enc, n_qubits, diag_n_features in specs_sorted:
        for a in ansatze_t:
            vals = got_express[enc][a]
            diag_rows.append(
                {
                    "encoding": enc,
                    "n_qubits": int(n_qubits),
                    "diag_n_features": int(diag_n_features),
                    "ansatz": a,
                    "kl_to_haar": vals["kl_to_haar"],
                    "mw_mean": vals["mw_mean"],
                    "mw_std": vals["mw_std"],
                }
            )

    grad_parts: list[pd.DataFrame] = []
    for enc, n_qubits, diag_n_features in specs_sorted:
        for a in ansatze_t:
            rows = []
            for d in sorted(depths_t):
                g = got_grad[(enc, a)][d]
                rows.append(
                    {
                        "ansatz": a,
                        "depth": int(d),
                        "n_trainable": -1,
                        "gradient_variance_mean": g["gradient_variance_mean"],
                        "gradient_variance_std": g["gradient_variance_std"],
                        "gradient_abs_mean": float("nan"),
                        "n_initializations": int(n_initializations),
                        "batch_size": int(batch_size),
                    }
                )
            gd = pd.DataFrame(rows)
            gd["encoding"] = enc
            gd["n_qubits"] = int(n_qubits)
            gd["diag_n_features"] = int(diag_n_features)
            grad_parts.append(gd)

    grad_df = pd.concat(grad_parts, ignore_index=True) if grad_parts else pd.DataFrame()
    return diag_rows, grad_df
