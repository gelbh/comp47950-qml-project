"""Section 10.4 device inference: submit or load cached per-anchor device results."""

from __future__ import annotations

import pickle
from collections.abc import Mapping
from typing import Any, NamedTuple, Protocol

import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from qml_project.ibm_runtime import (
    extract_counts_sequence_from_pub_result,
    run_ibm_sampler_pubs,
)
from qml_project.notebook_setup import workflow_cache_path, workflow_cache_search_dirs

from .qsvm import QSVMDevicePayload, build_qsvm_device_pubs, decode_qsvm_counts
from .vqc import VQCDevicePayload, build_vqc_device_pubs, decode_vqc_counts


class _TrainTestSplitLike(Protocol):
    X_test: Any
    y_test: Any


class DeviceInferenceSectionBundle(NamedTuple):
    """Outputs of :func:`run_device_inference_result_sweep` for Section 10.4 / 11."""

    device_results_by_pipeline: dict[str, pd.DataFrame]
    device_job_ids: dict[str, str]
    device_runtime_summaries: dict[str, dict[str, Any]]
    device_job_ids_by_size: dict[str, dict[int, str]]
    device_runtime_summaries_by_size: dict[str, dict[int, dict[str, Any]]]
    device_df: pd.DataFrame


def _submit_vqc(
    pipeline: str,
    payload: VQCDevicePayload,
    *,
    split: _TrainTestSplitLike,
    n_test: int,
    device_shots: int,
) -> tuple[pd.DataFrame, dict[str, Any], str | None]:
    X_test_subset = split.X_test[:n_test]
    y_test_subset = split.y_test[:n_test]
    pubs = build_vqc_device_pubs(payload, X_test_subset, shots=int(device_shots))
    result, summary = run_ibm_sampler_pubs(
        pubs,
        min_qubits=payload.n_qubits,
        enable_mitigation=False,
    )
    counts_list = extract_counts_sequence_from_pub_result(result[0])
    if len(counts_list) != len(X_test_subset):
        raise RuntimeError(
            f"[{pipeline}] expected {len(X_test_subset)} count rows from "
            f"VQC pub, got {len(counts_list)}."
        )
    y_pred = decode_vqc_counts(counts_list, payload)
    bacc = float(balanced_accuracy_score(y_test_subset, y_pred))
    train_size = int(getattr(payload, "train_size_used", 0) or 0)
    job_id = summary.get("job_id")
    print(
        f"[{pipeline}@n{train_size}] device run → balanced_accuracy={bacc:.4f} "
        f"on {len(y_test_subset)} samples "
        f"(backend={summary.get('backend_name')!r}, "
        f"quantum_seconds={summary.get('quantum_seconds')})"
    )
    quantum_seconds = float(summary.get("quantum_seconds") or 0.0)
    df = pd.DataFrame(
        [
            {
                "y_true": int(yt),
                "y_pred": int(yp),
                "balanced_accuracy": bacc,
                "n_test": int(len(y_test_subset)),
                "train_size": train_size,
                "inference_time_s": quantum_seconds,
                "shots": int(device_shots),
            }
            for yt, yp in zip(y_test_subset, y_pred, strict=True)
        ]
    )
    return df, dict(summary), str(job_id) if job_id else None


def _submit_qsvm(
    pipeline: str,
    payload: QSVMDevicePayload,
    *,
    split: _TrainTestSplitLike,
    n_test: int,
    device_shots: int,
    qsvm_pubs_per_job: int,
) -> tuple[pd.DataFrame, dict[str, Any], str | None]:
    X_test_subset = split.X_test[:n_test]
    y_test_subset = split.y_test[:n_test]
    pubs, pub_meta = build_qsvm_device_pubs(
        payload, X_test_subset, shots=int(device_shots)
    )
    n_qubits = int(pubs[0].circuit.num_qubits) if pubs else 1
    train_size = int(getattr(payload, "train_size_used", 0) or 0)
    print(
        f"[{pipeline}@n{train_size}] submitting {len(pubs)} overlap pubs in "
        f"chunks of {qsvm_pubs_per_job} (n_test={pub_meta['n_test']}, "
        f"n_sv={pub_meta['n_sv']}, n_qubits={n_qubits})"
    )
    all_counts: list[Mapping[str, int]] = []
    chunk_summaries: list[dict[str, Any]] = []
    for start in range(0, len(pubs), qsvm_pubs_per_job):
        chunk = pubs[start : start + qsvm_pubs_per_job]
        chunk_result, chunk_summary = run_ibm_sampler_pubs(
            chunk,
            min_qubits=n_qubits,
            enable_mitigation=False,
        )
        chunk_summaries.append(chunk_summary)
        for idx in range(len(chunk)):
            rows = extract_counts_sequence_from_pub_result(chunk_result[idx])
            if not rows:
                raise RuntimeError(
                    f"[{pipeline}] empty counts for overlap pub {start + idx}."
                )
            all_counts.append(rows[0])
    y_pred = decode_qsvm_counts(all_counts, payload, pub_meta)
    bacc = float(balanced_accuracy_score(y_test_subset, y_pred))
    total_quantum_seconds = sum(
        float(s.get("quantum_seconds") or 0.0) for s in chunk_summaries
    )
    first_backend = next(
        (s.get("backend_name") for s in chunk_summaries if s.get("backend_name")),
        None,
    )
    summary: dict[str, Any] = {
        "backend_name": first_backend,
        "quantum_seconds": total_quantum_seconds,
        "n_jobs": len(chunk_summaries),
        "job_ids": [s.get("job_id") for s in chunk_summaries if s.get("job_id")],
    }
    first_job_id = next(
        (s.get("job_id") for s in chunk_summaries if s.get("job_id")), None
    )
    print(
        f"[{pipeline}@n{train_size}] device run → balanced_accuracy={bacc:.4f} "
        f"on {len(y_test_subset)} samples "
        f"(backend={first_backend!r}, quantum_seconds={total_quantum_seconds:.2f}, "
        f"jobs={len(chunk_summaries)})"
    )
    df = pd.DataFrame(
        [
            {
                "y_true": int(yt),
                "y_pred": int(yp),
                "balanced_accuracy": bacc,
                "n_test": int(len(y_test_subset)),
                "train_size": train_size,
                "inference_time_s": float(total_quantum_seconds),
                "shots": int(device_shots),
            }
            for yt, yp in zip(y_test_subset, y_pred, strict=True)
        ]
    )
    return df, summary, str(first_job_id) if first_job_id else None


def load_disk_device_result_bundles(pipeline: str) -> dict[int, dict[str, Any]]:
    """Load §10 device-result pickles from every workflow-cache search directory.

    Only files named ``{pipeline}_device_result_n<size>.pkl`` with numeric
    ``<size>`` are loaded. Other filename shapes under the same glob are
    skipped. The first successful load per train size wins (search order from
    :func:`qml_project.notebook_setup.workflow_cache_search_dirs`).
    """
    out: dict[int, dict[str, Any]] = {}
    prefix = f"{pipeline}_device_result_n"
    for cache_dir in workflow_cache_search_dirs():
        for path in sorted(cache_dir.glob(f"{pipeline}_device_result_n*.pkl")):
            stem = path.stem
            if stem.endswith("_mit"):
                continue
            if not stem.startswith(prefix):
                continue
            size_str = stem[len(prefix) :]
            if not size_str.isdigit():
                continue
            n = int(size_str)
            if n in out:
                continue
            try:
                if path.stat().st_size == 0:
                    print(
                        f"[disk cache] skip empty file {path.name} "
                        f"(needs a successful RUN_DEVICE submission or a valid pickle)."
                    )
                    continue
                with path.open("rb") as fh:
                    bundle = pickle.load(fh)
            except Exception as exc:
                print(f"[disk cache] skip {path.name} (load failed: {exc})")
                continue
            if not isinstance(bundle, dict):
                print(f"[disk cache] skip {path.name} (expected dict bundle).")
                continue
            df = bundle.get("df")
            if (
                not isinstance(df, pd.DataFrame)
                or df.empty
                or "balanced_accuracy" not in df.columns
            ):
                print(
                    f"[disk cache] skip {path.name} "
                    f"(need non-empty df with balanced_accuracy)."
                )
                continue
            out[n] = bundle
            print(
                f"[disk cache] loaded {pipeline} n={n} ({len(df)} rows) from "
                f"{path.name} (dir={cache_dir})"
            )
    return out


def _write_result_cache(
    pipeline: str,
    train_size: int,
    df: pd.DataFrame,
    runtime_summary: dict[str, Any] | None,
    job_id: str | None,
) -> None:
    save_path = workflow_cache_path(f"{pipeline}_device_result_n{int(train_size)}.pkl")
    try:
        with save_path.open("wb") as fh:
            pickle.dump(
                {
                    "df": df,
                    "runtime_summary": runtime_summary,
                    "job_id": job_id,
                    "train_size": int(train_size),
                },
                fh,
            )
        print(f"[{pipeline}@n{train_size}] cached device result → {save_path.name}")
    except Exception as exc:
        print(f"[{pipeline}@n{train_size}] (cache write skipped: {exc})")


def run_device_inference_result_sweep(
    *,
    quantum_winners: Mapping[str, Any],
    device_payloads_by_pipeline: Mapping[str, Mapping[int, Any]],
    split: _TrainTestSplitLike,
    n_test: int,
    device_shots: int,
    run_device_flags: Mapping[str, bool],
    qsvm_pubs_per_job: int = 100,
    reuse_cached_device_results: bool = True,
    force_rerun_device_inference: bool = False,
) -> DeviceInferenceSectionBundle:
    """Run §10.4: load caches, optionally submit to IBM Runtime, aggregate ``device_df``."""
    device_results_by_pipeline: dict[str, pd.DataFrame] = {}
    device_job_ids: dict[str, str] = {}
    device_runtime_summaries: dict[str, dict[str, Any]] = {}
    device_job_ids_by_size: dict[str, dict[int, str]] = {}
    device_runtime_summaries_by_size: dict[str, dict[int, dict[str, Any]]] = {}

    if force_rerun_device_inference:
        print(
            "force_rerun_device_inference=True — skipping read from device-result "
            "caches; new Runtime jobs run where RUN_DEVICE_* is on, and successful "
            "runs still write pickles under .workflow_cache/."
        )

    cached_results: dict[str, dict[int, Any]] = {}
    for disk_pipeline in ("vqc", "qsvm"):
        cached_results[disk_pipeline] = load_disk_device_result_bundles(disk_pipeline)

    scan_dirs = workflow_cache_search_dirs()
    qsvm_pickles_per_dir: list[int] = []
    for d in scan_dirs:
        try:
            qsvm_pickles_per_dir.append(len(list(d.glob("qsvm_device_result_n*.pkl"))))
        except OSError:
            qsvm_pickles_per_dir.append(-1)
    print(
        f"[§10.4] workflow cache scan_dirs={len(scan_dirs)} "
        f"qsvm_device_result_n*.pkl per dir={qsvm_pickles_per_dir}\n"
        f"         primary_dir={scan_dirs[0] if scan_dirs else None}\n"
        f"         loaded train_sizes qsvm={sorted((cached_results.get('qsvm') or {}).keys())} "
        f"vqc={sorted((cached_results.get('vqc') or {}).keys())}"
    )

    for pipeline, _w in quantum_winners.items():
        run = run_device_flags.get(pipeline, False)
        per_pipeline_payloads = device_payloads_by_pipeline.get(pipeline) or {}
        per_pipeline_cache = cached_results.get(pipeline) or {}
        sizes = sorted(
            {int(s) for s in set(per_pipeline_payloads) | set(per_pipeline_cache)}
        )

        if not sizes:
            if run:
                raise RuntimeError(
                    f"[{pipeline}] no device payloads or cached results loaded — "
                    f"run Section 08 §8.5 to produce {pipeline}_device_payload_n<size>.pkl "
                    f"or place §10 `{pipeline}_device_result_n<size>.pkl` files."
                )
            print(
                f"[{pipeline}] RUN_DEVICE off — no §08 payloads and no readable "
                f"disk device-result pickles; empty device frame."
            )
            device_results_by_pipeline[pipeline] = pd.DataFrame()
            continue

        if not run:
            print(
                f"[{pipeline}] RUN_DEVICE off — IBM submission disabled; "
                f"using disk caches only (valid pickles rewritten in canonical form)."
            )

        per_size_frames: list[pd.DataFrame] = []
        device_job_ids_by_size.setdefault(pipeline, {})
        device_runtime_summaries_by_size.setdefault(pipeline, {})

        for size in sizes:
            size_i = int(size)
            use_disk_cache = not force_rerun_device_inference and (
                reuse_cached_device_results or not run
            )
            cached = per_pipeline_cache.get(size_i) if use_disk_cache else None
            if (
                cached is not None
                and isinstance(cached.get("df"), pd.DataFrame)
                and not cached["df"].empty
            ):
                df = cached["df"].copy()
                if "train_size" not in df.columns:
                    df["train_size"] = size_i
                if "balanced_accuracy" in df.columns:
                    _bacc_series = pd.to_numeric(
                        df["balanced_accuracy"], errors="coerce"
                    )
                    bacc_cached = float(_bacc_series.mean())
                else:
                    bacc_cached = None
                summary = (
                    dict(cached["runtime_summary"])
                    if cached.get("runtime_summary") is not None
                    else {}
                )
                job_id = str(cached["job_id"]) if cached.get("job_id") else None
                if summary:
                    device_runtime_summaries_by_size[pipeline][size_i] = summary
                if job_id:
                    device_job_ids_by_size[pipeline][size_i] = job_id
                print(
                    f"[{pipeline}@n{size_i}] reusing cached device result "
                    f"(rows={len(df)}, balanced_accuracy={bacc_cached}, "
                    f"job_id={job_id}) — no submission."
                )
                per_size_frames.append(df)
                if not run:
                    _write_result_cache(
                        pipeline,
                        size_i,
                        df,
                        summary or None,
                        job_id,
                    )
                continue

            payload = per_pipeline_payloads.get(size_i)
            if payload is None:
                if not run:
                    print(
                        f"[{pipeline}@n{size_i}] RUN_DEVICE off — no §08 payload "
                        f"for this anchor (skipped)."
                    )
                    continue
                raise RuntimeError(
                    f"[{pipeline}@n{size_i}] no device payload loaded — run Section "
                    f"08 §8.5 to produce {pipeline}_device_payload_n{size_i}.pkl."
                )

            if not run:
                print(
                    f"[{pipeline}@n{size_i}] RUN_DEVICE off — no valid disk cache "
                    f"for this anchor (skipped)."
                )
                continue

            if pipeline == "vqc":
                df, summary, job_id = _submit_vqc(
                    pipeline,
                    payload,
                    split=split,
                    n_test=n_test,
                    device_shots=device_shots,
                )
            elif pipeline == "qsvm":
                df, summary, job_id = _submit_qsvm(
                    pipeline,
                    payload,
                    split=split,
                    n_test=n_test,
                    device_shots=device_shots,
                    qsvm_pubs_per_job=qsvm_pubs_per_job,
                )
            else:
                print(
                    f"[{pipeline}@n{size_i}] unknown pipeline — emitting empty "
                    f"frame."
                )
                df, summary, job_id = pd.DataFrame(), {}, None

            if summary:
                device_runtime_summaries_by_size[pipeline][size_i] = summary
            if job_id:
                device_job_ids_by_size[pipeline][size_i] = job_id
            if isinstance(df, pd.DataFrame) and not df.empty:
                _write_result_cache(
                    pipeline,
                    size_i,
                    df,
                    summary or None,
                    job_id,
                )
                per_size_frames.append(df)

        if per_size_frames:
            device_results_by_pipeline[pipeline] = pd.concat(
                per_size_frames, ignore_index=True, sort=False
            )
        else:
            device_results_by_pipeline[pipeline] = pd.DataFrame()

        size_summaries = device_runtime_summaries_by_size.get(pipeline, {})
        if size_summaries:
            latest_size = max(size_summaries)
            device_runtime_summaries[pipeline] = dict(size_summaries[latest_size])
        size_jobs = device_job_ids_by_size.get(pipeline, {})
        if size_jobs:
            latest_size = max(size_jobs)
            device_job_ids[pipeline] = str(size_jobs[latest_size])

    non_empty_device: list[pd.DataFrame] = []
    for p, df in device_results_by_pipeline.items():
        if df.empty:
            continue
        tagged = df.copy()
        tagged["winner_pipeline"] = p
        non_empty_device.append(tagged)
    device_df = (
        pd.concat(non_empty_device, ignore_index=True, sort=False)
        if non_empty_device
        else pd.DataFrame()
    )

    return DeviceInferenceSectionBundle(
        device_results_by_pipeline=device_results_by_pipeline,
        device_job_ids=device_job_ids,
        device_runtime_summaries=device_runtime_summaries,
        device_job_ids_by_size=device_job_ids_by_size,
        device_runtime_summaries_by_size=device_runtime_summaries_by_size,
        device_df=device_df,
    )
