"""Results page: combined selection table, cost vs accuracy scatter, three-way bar."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _classical_configs_like_selection(classical_df: pd.DataFrame | None) -> pd.DataFrame:
    """One row per (model, feature_set, symmetry) at max train_size, OOD — mirrors Section 07."""
    if classical_df is None or classical_df.empty:
        return pd.DataFrame()
    sub = classical_df.copy()
    if "sub_study" in sub.columns:
        sub = sub.loc[sub["sub_study"].astype(str) == "main"]
    if "regime" in sub.columns and (sub["regime"] == "ood").any():
        sub = sub.loc[sub["regime"] == "ood"]
    if sub.empty or "balanced_accuracy" not in sub.columns:
        return pd.DataFrame()
    if "train_size" not in sub.columns:
        return pd.DataFrame()
    sub["train_size"] = pd.to_numeric(sub["train_size"], errors="coerce")
    max_n = sub["train_size"].max()
    if pd.isna(max_n):
        return pd.DataFrame()
    sub = sub.loc[sub["train_size"] == max_n].copy()
    gcols = [c for c in ("model", "feature_set", "symmetry") if c in sub.columns]
    if not gcols:
        return pd.DataFrame()
    agg: dict[str, Any] = {
        "mean_accuracy": ("balanced_accuracy", "mean"),
        "std_accuracy": ("balanced_accuracy", "std"),
    }
    if "train_time_s" in sub.columns:
        agg["mean_cost"] = ("train_time_s", "mean")
    grouped = sub.groupby(gcols, dropna=False).agg(**agg).reset_index()
    grouped["pipeline"] = "classical"
    grouped["selection_id"] = grouped[gcols].astype(str).agg("|".join, axis=1)
    grouped["encoding"] = grouped["feature_set"] if "feature_set" in grouped.columns else None
    grouped["include_nim_sum"] = np.nan
    grouped["pareto"] = False
    grouped["winner"] = False
    if "mean_cost" not in grouped.columns:
        grouped["mean_cost"] = np.nan
    grouped["train_size_used"] = int(max_n)
    grouped["tier"] = "sim"
    return grouped


def _normalize_include_nim_sum(val: Any) -> Any:
    """Coerce workflow / parquet values to bool; non-boolean strings → NaN."""
    if val is None or val is pd.NA:
        return np.nan
    if pd.api.types.is_scalar(val) and pd.isna(val):
        return np.nan
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (int, np.integer)) and val in (0, 1):
        return bool(int(val))
    s = str(val).strip().lower()
    if s in ("true", "1", "t", "yes"):
        return True
    if s in ("false", "0", "f", "no"):
        return False
    return np.nan


def _uniq_include_nim_sum_or_nan(series: pd.Series) -> Any:
    """One bool if the group agrees; else NaN (e.g. mixed sweep keys on same label)."""
    u = series.dropna().map(_normalize_include_nim_sum).dropna().unique()
    if len(u) == 0:
        return np.nan
    if len(u) > 1:
        return np.nan
    return bool(u[0])


def _vqc_config_id_to_include_nim_sum(vqc_workflow_df: pd.DataFrame) -> pd.Series:
    """Map ``config_id`` → ``include_nim_sum`` from Section 05 workflow rows."""
    need = ("config_id", "include_nim_sum")
    if vqc_workflow_df.empty or not all(c in vqc_workflow_df.columns for c in need):
        return pd.Series(dtype=object)
    sub = vqc_workflow_df.loc[:, list(need)].dropna(subset=["config_id"]).copy()
    sub["config_id"] = sub["config_id"].astype(str)
    sub = sub.drop_duplicates(subset=["config_id"], keep="first")
    ser = sub.set_index("config_id")["include_nim_sum"].map(_normalize_include_nim_sum)
    return ser


def _enrich_include_nim_sum_from_workflows(
    frame: pd.DataFrame,
    *,
    vqc_workflow_df: pd.DataFrame | None,
    qsvm_workflow_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Fill missing ``include_nim_sum`` on simulation VQC/QSVM rows from workflow parquets."""
    out = frame.copy()
    if "include_nim_sum" not in out.columns:
        out["include_nim_sum"] = np.nan
    tier = out.get("tier", pd.Series("sim", index=out.index)).fillna("sim").astype(str)
    pipe = out["pipeline"].astype(str).str.lower()
    sim = tier != "device"

    if vqc_workflow_df is not None and not vqc_workflow_df.empty:
        lut = _vqc_config_id_to_include_nim_sum(vqc_workflow_df)
        if not lut.empty:
            m = sim & (pipe == "vqc") & out["include_nim_sum"].isna()
            if m.any():
                if "config_id" in out.columns:
                    key = out["config_id"].where(
                        out["config_id"].notna()
                        & (out["config_id"].astype(str).str.len() > 0),
                        np.nan,
                    )
                else:
                    key = pd.Series(np.nan, index=out.index)
                if "selection_id" in out.columns:
                    key = key.fillna(out["selection_id"])
                key = key.astype(str)
                mapped = key.map(lut)
                fill = m & mapped.notna()
                out.loc[fill, "include_nim_sum"] = mapped.loc[fill]

    if qsvm_workflow_df is not None and not qsvm_workflow_df.empty:
        cols = ("variant_id", "encoding", "include_nim_sum")
        if all(c in qsvm_workflow_df.columns for c in cols):
            enrich = (
                qsvm_workflow_df.loc[:, list(cols)]
                .dropna(subset=["variant_id", "encoding"])
                .groupby(["variant_id", "encoding"], dropna=False)["include_nim_sum"]
                .agg(_uniq_include_nim_sum_or_nan)
                .reset_index(name="_nim_wf")
            )
            m = sim & (pipe == "qsvm") & out["include_nim_sum"].isna()
            if m.any() and "variant_id" in out.columns and "encoding" in out.columns:
                hit = (
                    out.loc[m, ["variant_id", "encoding"]]
                    .reset_index(drop=True)
                    .merge(enrich, on=["variant_id", "encoding"], how="left")
                )
                nim_wf = hit["_nim_wf"]
                idx = out.index[m]
                out.loc[idx, "include_nim_sum"] = nim_wf.map(_normalize_include_nim_sum).to_numpy()

    return out


def build_combined_selection_view(
    selection_df: pd.DataFrame | None,
    classical_df: pd.DataFrame | None,
    device_df: pd.DataFrame | None = None,
    *,
    vqc_workflow_df: pd.DataFrame | None = None,
    qsvm_workflow_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Concatenate Section 07 quantum rows, classical summaries, and Section 10 device rows.

    When ``vqc_workflow_df`` / ``qsvm_workflow_df`` are provided (same Section 05/06
    parquets as the notebooks), missing ``include_nim_sum`` on simulation quantum
    rows is filled from workflow columns so the Results scatter can split Nim-sum
    on vs off without relying on id-string parsing alone.
    """
    parts: list[pd.DataFrame] = []
    if selection_df is not None and not selection_df.empty:
        sel = selection_df.copy()
        sel["tier"] = "sim"
        parts.append(sel)
    classical_part = _classical_configs_like_selection(classical_df)
    if not classical_part.empty:
        parts.append(classical_part)
    if device_df is not None and not device_df.empty:
        parts.append(device_df.copy())
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    if "tier" not in out.columns:
        out["tier"] = "sim"
    else:
        out["tier"] = out["tier"].fillna("sim").astype(str)
    out = _enrich_include_nim_sum_from_workflows(
        out,
        vqc_workflow_df=vqc_workflow_df,
        qsvm_workflow_df=qsvm_workflow_df,
    )
    return out


def _quantum_encoding_channel(row: pd.Series, pipe: str) -> str:
    """Classify quantum sim rows for raw vs Nim-sum in encoding (see Sections 05/06).

    Ambiguous ids (e.g. missing ``|ns=`` or aggregated ``ns=(T,F)``) map to
    ``raw_enc`` so the Results scatter never invents a third Nim-sum legend.
    """
    if pipe not in ("vqc", "qsvm"):
        return "raw_enc"
    if "include_nim_sum" in row.index and pd.notna(row["include_nim_sum"]):
        try:
            return "nim_enc" if bool(row["include_nim_sum"]) else "raw_enc"
        except (TypeError, ValueError):
            pass
    blob = "|".join(
        str(row.get(k) or "")
        for k in ("variant_id", "selection_id", "config_id")
    ).lower()
    if "|ns=f" in blob or blob.endswith("ns=f") or "ns=false" in blob.replace(" ", ""):
        return "raw_enc"
    if "|ns=t" in blob or blob.endswith("ns=t") or "ns=true" in blob.replace(" ", ""):
        return "nim_enc"
    return "raw_enc"


def _scatter_channel_key(row: pd.Series) -> str:
    """Stable legend key: classical raw/parity, quantum raw_enc/nim_enc, device."""
    tier = str(row.get("tier", "sim"))
    pipe = str(row.get("pipeline", "")).lower()
    if tier == "device":
        return f"dev_{pipe}"
    if pipe == "classical":
        fs = str(row.get("feature_set", "")).lower()
        return "cl_raw" if fs == "raw" else "cl_parity"
    if pipe in ("vqc", "qsvm"):
        ch = _quantum_encoding_channel(row, pipe)
        return f"{pipe}_{ch}"
    return f"other_{pipe}"


_SCATTER_CHANNEL_STYLE: dict[str, tuple[str, str, str, int]] = {
    "cl_raw": ("Classical (raw heaps)", "#e9c46a", "circle", 8),
    "cl_parity": ("Classical (parity / engineered)", "#bc8c2f", "diamond", 8),
    "vqc_raw_enc": ("VQC sim — no Nim-sum in encoding", "#264653", "circle", 8),
    "vqc_nim_enc": ("VQC sim — Nim-sum in encoding", "#264653", "square", 8),
    "qsvm_raw_enc": ("QSVM sim — no Nim-sum in encoding", "#2a9d8f", "circle", 8),
    "qsvm_nim_enc": ("QSVM sim — Nim-sum in encoding", "#2a9d8f", "square", 8),
    "dev_vqc": ("VQC (IBM device)", "#264653", "star", 11),
    "dev_qsvm": ("QSVM (IBM device)", "#2a9d8f", "star", 11),
}


def combined_selection_cost_scatter(combined: pd.DataFrame) -> go.Figure:
    """OOD BA vs seconds: simulation = mean training time; IBM device = runtime summary."""
    fig = go.Figure()
    if combined is None or combined.empty or "mean_accuracy" not in combined.columns:
        fig.update_layout(
            title="Time vs OOD balanced accuracy",
            height=280,
            annotations=[
                dict(
                    text="No rows to plot.",
                    showarrow=False,
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                )
            ],
        )
        return fig

    sub = combined.copy()
    sub["tier"] = sub.get("tier", pd.Series(["sim"] * len(sub))).fillna("sim").astype(str)
    y = pd.to_numeric(sub["mean_accuracy"], errors="coerce")
    x = pd.to_numeric(sub["mean_cost"], errors="coerce") if "mean_cost" in sub.columns else pd.Series(np.nan, index=sub.index)

    x_title = "Seconds (log): sim = mean training time · IBM = device runtime"
    pos = x[x > 0]
    use_log = bool(len(pos) > 1 and float(pos.max() / pos.min()) > 50)
    if use_log:
        x = x.clip(lower=1e-12)

    if "pipeline" not in sub.columns:
        sub["pipeline"] = ""

    def _hover_block(r: pd.Series, pipe: str, tier: str) -> str:
        parts = [f"pipeline={pipe}", f"tier={tier}"]
        for key in (
            "selection_id",
            "encoding",
            "include_nim_sum",
            "model",
            "feature_set",
            "symmetry",
            "mean_accuracy",
            "mean_cost",
            "train_size",
            "pareto",
            "winner",
        ):
            if key in sub.columns and pd.notna(r.get(key)):
                parts.append(f"{key}={r[key]}")
        return "<br>".join(parts)

    def _scatter_style(chan: str) -> tuple[str, str, str, int]:
        return _SCATTER_CHANNEL_STYLE.get(
            chan,
            (chan.replace("other_", "").replace("_", " "), "#888888", "circle", 8),
        )

    sub["_scatter_chan"] = sub.apply(_scatter_channel_key, axis=1)
    chan_order = [
        "cl_raw",
        "cl_parity",
        "vqc_raw_enc",
        "vqc_nim_enc",
        "qsvm_raw_enc",
        "qsvm_nim_enc",
        "dev_vqc",
        "dev_qsvm",
    ]
    present = set(sub["_scatter_chan"].dropna().astype(str))
    ordered_chans = [c for c in chan_order if c in present] + sorted(
        c for c in present if c not in chan_order
    )

    for chan in ordered_chans:
        m = sub["_scatter_chan"] == chan
        if not m.any():
            continue
        sub_m = sub.loc[m]
        xs_m = x.reindex(sub_m.index)
        ys_m = y.reindex(sub_m.index)
        fin = xs_m.notna() & ys_m.notna()
        sub_plot = sub_m.loc[fin]
        xs_plot = xs_m.loc[fin]
        ys_plot = ys_m.loc[fin]
        if len(xs_plot) == 0:
            continue
        trace_name, col, sym, base_size = _scatter_style(chan)
        is_device = chan.startswith("dev_")
        hovers = [
            _hover_block(
                r,
                str(r.get("pipeline", "")).lower(),
                str(r.get("tier", "sim")),
            )
            for _, r in sub_plot.iterrows()
        ]
        sizes: list[int] = []
        for _, r in sub_plot.iterrows():
            if is_device:
                sizes.append(11)
            else:
                w = r.get("winner") if "winner" in sub.columns else False
                try:
                    sizes.append(14 if bool(w) else base_size)
                except (TypeError, ValueError):
                    sizes.append(base_size)
        fig.add_trace(
            go.Scatter(
                x=xs_plot,
                y=ys_plot,
                mode="markers",
                name=trace_name,
                marker=dict(
                    size=sizes,
                    symbol=sym,
                    color=col,
                    line=dict(width=0.5, color="#222"),
                ),
                text=hovers,
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title="Seconds vs OOD BA — raw vs parity (classical) · Nim-sum on/off (quantum sim)",
        xaxis_title=x_title,
        yaxis_title="mean OOD balanced accuracy",
        yaxis=dict(range=[0, 1.05]),
        height=440,
        margin=dict(l=48, r=20, t=52, b=96),
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.42, x=0.5, xanchor="center"),
    )
    if use_log:
        fig.update_xaxes(type="log", exponentformat="power", showexponent="all")
    if not fig.data:
        fig.update_layout(
            annotations=[
                dict(
                    text="No points with finite x (training time or device runtime).",
                    showarrow=False,
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                )
            ],
        )
    return fig


def selection_table_scatter(df: pd.DataFrame) -> go.Figure:
    """Backward-compatible single scatter (quantum selection rows only)."""
    return combined_selection_cost_scatter(df)


def three_way_results_bar(
    classical_df: pd.DataFrame | None,
    vqc_winners: pd.DataFrame | None,
    qsvm_winners: pd.DataFrame | None,
    *,
    metric: str = "balanced_accuracy",
) -> go.Figure:
    """Compact three-pipeline bar of best observed mean balanced accuracy.

    For each pipeline we take the single strongest result we can find in
    its cached parquet. Missing pipelines are dropped rather than shown
    at zero so the audience does not misread the chart.
    """
    fig = go.Figure()
    rows: list[dict] = []

    def _best_from_df(df: pd.DataFrame | None, label: str, color: str) -> None:
        if df is None or df.empty or metric not in df.columns:
            return
        sub = df.copy()
        if "regime" in sub.columns and (sub["regime"] == "ood").any():
            sub = sub[sub["regime"] == "ood"]
        group_keys = [
            c for c in ("model", "feature_set", "encoding", "config_id") if c in sub.columns
        ]
        if not group_keys:
            value = float(sub[metric].mean())
        else:
            grouped = sub.groupby(group_keys)[metric].mean()
            value = float(grouped.max())
        rows.append(dict(pipeline=label, value=value, color=color))

    _best_from_df(classical_df, "Classical", "#e9c46a")
    _best_from_df(vqc_winners, "VQC", "#264653")
    _best_from_df(qsvm_winners, "QSVM", "#2a9d8f")

    if not rows:
        fig.update_layout(
            title="Best OOD balanced accuracy — three-way",
            annotations=[
                dict(
                    text="No result parquets found.",
                    showarrow=False,
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                )
            ],
            height=260,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        return fig

    fig.add_trace(
        go.Bar(
            x=[r["pipeline"] for r in rows],
            y=[r["value"] for r in rows],
            marker_color=[r["color"] for r in rows],
            text=[f"{r['value']:.2f}" for r in rows],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Best OOD balanced accuracy — classical vs VQC vs QSVM",
        yaxis=dict(range=[0, 1.05], title=metric.replace("_", " ")),
        xaxis_title="pipeline",
        margin=dict(l=40, r=20, t=50, b=40),
        height=320,
        plot_bgcolor="white",
        showlegend=False,
    )
    return fig
