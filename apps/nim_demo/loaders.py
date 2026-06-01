"""Cached loaders for trained models used by the interactive demo.

VQC/QSVM device-ready payloads come from notebook Section 08 (and may be
refreshed after Section 10 inference). Files live under
``notebooks/.workflow_cache/``; loading stays sub-second on startup.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import streamlit as st

from qml_project.baselines.features import prepare_features
from qml_project.baselines.models import create_models
from qml_project.device_inference import QSVMDevicePayload, VQCDevicePayload
from qml_project.nim.data import prepare_experiment_data

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def project_root() -> Path:
    """Return the repository root (three levels up from this file)."""
    return Path(__file__).resolve().parents[2]


def workflow_cache_dir() -> Path:
    return project_root() / "notebooks" / ".workflow_cache"


def _list_payload_train_sizes(*, glob_pattern: str) -> list[int]:
    """Parse ``..._n{train}.pkl`` stems under the workflow cache."""
    sizes: list[int] = []
    for p in workflow_cache_dir().glob(glob_pattern):
        try:
            n = int(p.stem.rsplit("_n", 1)[1])
        except ValueError:
            continue
        sizes.append(n)
    return sorted(sizes)


def list_vqc_payload_sizes() -> list[int]:
    """Return sorted train sizes for which a VQC payload pickle exists."""
    return _list_payload_train_sizes(glob_pattern="vqc_device_payload_n*.pkl")


def list_qsvm_payload_sizes() -> list[int]:
    """Return sorted train sizes for which a QSVM payload pickle exists."""
    return _list_payload_train_sizes(glob_pattern="qsvm_device_payload_n*.pkl")


# ---------------------------------------------------------------------------
# Classical bundle
# ---------------------------------------------------------------------------


@dataclass
class ClassicalBundle:
    """Fitted classical baseline wrapped with its feature transform."""

    name: str
    feature_set: str
    model: Any
    feature_names: list[str]
    train_balanced_accuracy: float
    test_balanced_accuracy: float
    M: int = 7


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def load_vqc_payload(train_size: int) -> VQCDevicePayload:
    """Load a ``VQCDevicePayload`` pickled during Section 08 (pre-device validation)."""
    path = workflow_cache_dir() / f"vqc_device_payload_n{train_size}.pkl"
    with path.open("rb") as fh:
        payload = pickle.load(fh)
    if not isinstance(payload, VQCDevicePayload):
        raise TypeError(
            f"Expected VQCDevicePayload, got {type(payload).__name__} from {path}"
        )
    return payload


@st.cache_resource(show_spinner=False)
def load_qsvm_payload(train_size: int) -> QSVMDevicePayload:
    """Load a ``QSVMDevicePayload`` pickled during Section 08 (pre-device validation)."""
    path = workflow_cache_dir() / f"qsvm_device_payload_n{train_size}.pkl"
    with path.open("rb") as fh:
        payload = pickle.load(fh)
    if not isinstance(payload, QSVMDevicePayload):
        raise TypeError(
            f"Expected QSVMDevicePayload, got {type(payload).__name__} from {path}"
        )
    return payload


@st.cache_resource(show_spinner=False)
def load_classical_bundle(
    *,
    model_name: str = "Logistic Regression",
    feature_set: str = "parity",
    M: int = 7,
    random_state: int = 42,
) -> ClassicalBundle:
    """Fit a fresh classical baseline on the OOD training split.

    Training on the enumerated training states (heaps ≤ 5) takes well
    under a second for any of the ``create_models()`` estimators.
    """
    from sklearn.metrics import balanced_accuracy_score

    data = prepare_experiment_data(k=3, M=M, random_state=random_state)
    X_train = prepare_features(data.split.X_train, feature_set=feature_set, M=M)
    X_test = prepare_features(data.split.X_test, feature_set=feature_set, M=M)

    models = create_models(random_state=random_state, M=M)
    if model_name not in models:
        raise KeyError(f"Unknown classical model {model_name!r}; choose from {list(models)}")
    model = models[model_name]
    model.fit(X_train, data.split.y_train)

    train_ba = float(
        balanced_accuracy_score(data.split.y_train, model.predict(X_train))
    )
    test_ba = float(balanced_accuracy_score(data.split.y_test, model.predict(X_test)))

    feature_names = classical_feature_component_names(
        feature_set=feature_set, M=M, k=3
    )
    return ClassicalBundle(
        name=model_name,
        feature_set=feature_set,
        model=model,
        feature_names=feature_names,
        train_balanced_accuracy=train_ba,
        test_balanced_accuracy=test_ba,
        M=M,
    )


# Parquet names that ``load_summary_dataframes`` will try to load.
_SUMMARY_PARQUET_NAMES: tuple[str, ...] = (
    "classical_df",
    "vqc_workflow_df",
    "qsvm_workflow_df",
    "quantum_winner_rows_vqc",
    "quantum_winner_rows_qsvm",
    "quantum_winners_summary",
    "selection_table",
)


@st.cache_data(show_spinner=False)
def load_summary_dataframes() -> dict[str, pd.DataFrame]:
    """Return optional summary parquets from ``notebooks/.workflow_cache/`` if present.

    Missing files are silently skipped so the demo still renders on a
    machine that has not run the full workflow. The returned dict maps
    the parquet stem (e.g. ``"classical_df"``) to its DataFrame.
    """
    out: dict[str, pd.DataFrame] = {}
    cache = workflow_cache_dir()
    for name in _SUMMARY_PARQUET_NAMES:
        path = cache / f"{name}.parquet"
        if not path.exists():
            continue
        try:
            out[name] = pd.read_parquet(path)
        except Exception:
            continue
    return out


def _device_result_rows_for_pipeline(
    cache: Path,
    *,
    pipeline: Literal["vqc", "qsvm"],
    glob_pattern: str,
    stem_prefix: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(cache.glob(glob_pattern)):
        if path.stem.endswith("_mit"):
            continue
        size_str = path.stem.removeprefix(stem_prefix)
        try:
            n = int(size_str)
        except ValueError:
            continue
        try:
            with path.open("rb") as fh:
                bundle = pickle.load(fh)
        except Exception:
            continue
        ba = _extract_balanced_accuracy(bundle)
        if ba is None:
            continue
        rt = _extract_runtime_summary_seconds(bundle)
        rows.append(
            {
                "pipeline": pipeline,
                "train_size": n,
                "balanced_accuracy": ba,
                "mean_accuracy": float(ba),
                "mean_cost": float(rt) if rt is not None else np.nan,
                "tier": "device",
                "selection_id": f"device|{pipeline}|n={n}",
                "encoding": "ibm_device",
                "include_nim_sum": np.nan,
                "pareto": False,
                "winner": False,
                "std_accuracy": np.nan,
                "train_size_used": n,
            }
        )
    return rows


@st.cache_resource(show_spinner=False)
def load_device_history() -> pd.DataFrame | None:
    """Return Section 10 on-device rows for scatter plots (BA + optional runtime).

    Loads only **unmitigated** caches (``*_device_result_n<size>.pkl``).
    Mitigated ``*_mit.pkl`` files are skipped — in our runs they matched the
    vanilla numbers, so the demo omits them to avoid duplicate bars.

    Columns match the combined Results scatter contract: ``pipeline`` is
    lowercase ``vqc`` / ``qsvm``, ``mean_accuracy`` is OOD balanced accuracy,
    ``mean_cost`` is seconds from ``runtime_summary`` when present (for the
    time scatter x-axis), ``tier`` is ``device``, ``selection_id`` tags the run.
    """
    rows: list[dict[str, Any]] = []
    cache = workflow_cache_dir()
    for pipeline, glob_pat, stem_prefix in (
        ("vqc", "vqc_device_result_n*.pkl", "vqc_device_result_n"),
        ("qsvm", "qsvm_device_result_n*.pkl", "qsvm_device_result_n"),
    ):
        rows.extend(
            _device_result_rows_for_pipeline(
                cache, pipeline=pipeline, glob_pattern=glob_pat, stem_prefix=stem_prefix
            )
        )
    if not rows:
        return None
    return pd.DataFrame(rows).sort_values(["pipeline", "train_size"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def classical_feature_component_names(
    *, feature_set: str, M: int = 7, k: int = 3
) -> list[str]:
    """Human-readable names for each column of ``prepare_features`` (same order)."""
    return _classical_feature_names(feature_set=feature_set, M=M, k=k)


def _classical_feature_names(*, feature_set: str, M: int, k: int) -> list[str]:
    """Human-readable names for each feature in ``prepare_features``'s output."""
    raw = [f"h{i + 1}/M" for i in range(k)]
    parity = [f"h{i + 1} mod 2" for i in range(k)]
    pairs = [
        f"(h{i + 1}\u2295h{j + 1})/M"
        for i in range(k)
        for j in range(i + 1, k)
    ]
    n_bits = int(np.ceil(np.log2(M + 1))) if M > 0 else 0
    bit_parities = [f"bit{b}-parity" for b in range(n_bits)]
    if feature_set == "raw":
        return raw
    if feature_set == "heap_parity":
        return raw + parity
    if feature_set == "pairwise_xor":
        return raw + pairs
    if feature_set == "bit_parity":
        return raw + bit_parities
    if feature_set == "parity":
        return raw + parity + pairs + bit_parities
    return [f"f{i}" for i in range(len(raw))]


def _extract_runtime_summary_seconds(bundle: Any) -> float | None:
    """Best-effort wall/quantum seconds from Section 10 ``runtime_summary`` dict."""
    if not isinstance(bundle, dict):
        return None
    summary = bundle.get("runtime_summary")
    if not isinstance(summary, dict):
        return None
    for key in (
        "wall_seconds",
        "total_seconds",
        "quantum_seconds",
        "job_runtime_seconds",
        "usage_seconds",
    ):
        val = summary.get(key)
        if val is None:
            continue
        try:
            out = float(val)
        except (TypeError, ValueError):
            continue
        if out > 0:
            return out
    return None


def _extract_balanced_accuracy(bundle: Any) -> float | None:
    """Best-effort extraction of a balanced accuracy from a pickled result.

    Section 10 caches ``{"df": DataFrame, "runtime_summary": ..., ...}`` where
    ``balanced_accuracy`` lives on each row of ``df`` (same value repeated),
    not at the top level of the dict.
    """
    if isinstance(bundle, dict):
        for key in ("balanced_accuracy", "ba", "bal_acc"):
            if key in bundle:
                try:
                    return float(bundle[key])
                except (TypeError, ValueError):
                    return None
        df = bundle.get("df")
        if isinstance(df, pd.DataFrame) and "balanced_accuracy" in df.columns:
            col = pd.to_numeric(df["balanced_accuracy"], errors="coerce")
            if col.notna().any():
                return float(col.mean())
        metrics = bundle.get("metrics")
        if isinstance(metrics, dict) and "balanced_accuracy" in metrics:
            try:
                return float(metrics["balanced_accuracy"])
            except (TypeError, ValueError):
                return None
    ba = getattr(bundle, "balanced_accuracy", None)
    if ba is not None:
        try:
            return float(ba)
        except (TypeError, ValueError):
            return None
    return None
