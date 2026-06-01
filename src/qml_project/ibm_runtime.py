"""IBM Quantum Runtime helpers for ``SamplerV2`` (inference jobs).

``qiskit_ibm_runtime`` is imported only inside functions that contact the
cloud, so ``import qml_project`` still works in simulation-only environments.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence, cast

import numpy as np
from qiskit.primitives.containers.sampler_pub import SamplerPub
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qml_project.training.noise_aer import (
    build_assignment_matrix_from_symmetric_readout_error,
)
from tenacity import retry, stop_after_attempt, wait_exponential

_RUNTIME_WAIT = wait_exponential(multiplier=1, min=1, max=45)
_METRICS_WAIT = wait_exponential(multiplier=1, min=0.5, max=20)


def _runtime_job_result(job: Any) -> Any:
    """Blocking ``job.result()`` with bounded retries on transient Runtime/API faults."""

    @retry(stop=stop_after_attempt(5), wait=_RUNTIME_WAIT)
    def _pull() -> Any:
        return job.result()

    return _pull()


def _job_metrics_dict(job: Any) -> dict[str, Any]:
    """Fetch ``job.metrics()`` with bounded retries (then fall back like the legacy path)."""

    @retry(stop=stop_after_attempt(4), wait=_METRICS_WAIT)
    def _pull() -> dict[str, Any]:
        return cast(dict[str, Any], job.metrics())

    try:
        return _pull()
    except Exception:
        return {}


def _job_usage_quantum_seconds(job: Any) -> float | None:
    """Best-effort ``job.usage()`` as quantum-seconds with retries."""

    @retry(stop=stop_after_attempt(4), wait=_METRICS_WAIT)
    def _pull() -> float | None:
        usage = job.usage()
        return float(usage) if usage is not None else None

    try:
        return _pull()
    except Exception:
        return None


def _parse_iso_datetime(value: Any) -> datetime | None:
    """Parse timestamp strings from ``job.metrics()`` ``timestamps`` payload."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc)
    if not isinstance(value, str):
        return None
    text = value.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _delta_seconds(
    start: datetime | None, end: datetime | None
) -> float | None:
    if start is None or end is None:
        return None
    return float((end - start).total_seconds())


def _extract_counts_from_pub_result(pub_result: Any) -> dict[str, int]:
    """Extract counts from a SamplerV2 pub result across register layouts."""
    try:
        joined = pub_result.join_data()
        return dict(joined.get_counts())
    except Exception:
        pass

    data = getattr(pub_result, "data", None)
    if data is None:
        return {}

    # Fallback for older/newer layouts: first register-like entry with get_counts.
    try:
        values = list(data.values())
    except Exception:
        values = [getattr(data, name) for name in data.keys()]
    for value in values:
        getter = getattr(value, "get_counts", None)
        if callable(getter):
            raw = cast(Mapping[str, int], getter())
            return dict(raw)
    return {}


def _extract_counts_sequence_from_getter(
    getter: Any,
    *,
    max_items: int = 100_000,
) -> list[dict[str, int]]:
    """Collect one or more count dictionaries from a ``get_counts`` callable.

    Tries indexed access first: for a V2 ``BitArray`` with shape ``(N,)``,
    ``get_counts(i)`` returns per-binding counts while ``get_counts()`` with
    no argument returns the **aggregate** across the full shape — which
    silently collapses batched V2 pubs into a single count dict. Indexed
    access therefore has to win before we fall back to the aggregate form.
    """
    if not callable(getter):
        return []

    rows: list[dict[str, int]] = []
    for i in range(max_items):
        try:
            item = getter(i)
        except (TypeError, IndexError, KeyError, StopIteration, ValueError):
            break
        except Exception:
            break
        if not isinstance(item, Mapping):
            break
        rows.append(dict(cast(Mapping[str, int], item)))
    if rows:
        return rows

    # Fall back: no-arg form for containers that only expose aggregate counts
    # or a pre-materialised list/tuple of per-binding dicts.
    try:
        raw = getter()
    except Exception:
        return []
    if isinstance(raw, Mapping):
        return [dict(cast(Mapping[str, int], raw))]
    if isinstance(raw, (list, tuple)):
        return [
            dict(cast(Mapping[str, int], x))
            for x in raw
            if isinstance(x, Mapping)
        ]
    return []


def extract_counts_sequence_from_pub_result(pub_result: Any) -> list[dict[str, int]]:
    """Extract per-sample counts from a Runtime/Aer Sampler V2 pub result."""
    data = getattr(pub_result, "data", None)
    if data is None:
        one = _extract_counts_from_pub_result(pub_result)
        return [one] if one else []

    meas = getattr(data, "meas", None)
    rows = _extract_counts_sequence_from_getter(getattr(meas, "get_counts", None))
    if rows:
        return rows

    # Fallback for alternate register names/layouts.
    values: list[Any] = []
    try:
        values = list(data.values())
    except Exception:
        try:
            keys = list(data.keys())
            values = [getattr(data, str(k)) for k in keys]
        except Exception:
            values = []
    for value in values:
        rows = _extract_counts_sequence_from_getter(getattr(value, "get_counts", None))
        if rows:
            return rows

    one = _extract_counts_from_pub_result(pub_result)
    return [one] if one else []


def extract_counts_sequence_from_sampler_result(result: Any) -> list[dict[str, int]]:
    """Extract per-sample counts for the first PUB in a sampler result payload."""
    try:
        pub0 = result[0]
    except Exception:
        return []
    return extract_counts_sequence_from_pub_result(pub0)


def summarize_runtime_job(job: Any) -> dict[str, Any]:
    """Return backend, queue and usage diagnostics for a Runtime job."""
    metrics = _job_metrics_dict(job)
    quantum_seconds = _job_usage_quantum_seconds(job)

    ts = metrics.get("timestamps") or {}
    t_created = _parse_iso_datetime(ts.get("created"))
    t_running = _parse_iso_datetime(ts.get("running"))
    t_ended = _parse_iso_datetime(
        ts.get("ended") or ts.get("completed") or ts.get("finished")
    )
    queue_wait_seconds = _delta_seconds(t_created, t_running)
    running_seconds = _delta_seconds(t_running, t_ended)
    return {
        "job_id": getattr(job, "job_id", lambda: None)(),
        "quantum_seconds": quantum_seconds,
        "queue_wait_seconds": queue_wait_seconds,
        "running_seconds": running_seconds,
        "metrics": metrics,
    }


def run_ibm_sampler_pubs(
    pubs: Sequence[SamplerPub],
    *,
    service: Any | None = None,
    backend: Any | None = None,
    min_qubits: int = 1,
    enable_mitigation: bool = False,
    dd_sequence: str = "XpXm",
) -> tuple[Any, dict[str, Any]]:
    """Run one or more PUBs on IBM Runtime SamplerV2 and return diagnostics.

    When ``enable_mitigation=True``, turn on two low-overhead sampler-side
    error-reduction features supported natively by the Runtime V2 primitive:

    - **TREX (Twirled Readout Error eXtinction)** via
      ``options.twirling.enable_measure = True``. Pauli-twirls the
      measurement so readout bias averages out. Typical benefit on
      7-qubit circuits with ~1–2 %/qubit assignment error is ~0.05–0.15
      balanced-accuracy uplift. Cost: negligible in quantum-seconds
      because shots are redistributed across randomisations rather than
      multiplied.
    - **Dynamical decoupling** via
      ``options.dynamical_decoupling.enable = True`` with ``dd_sequence``
      (default ``"XpXm"``). Protects idle qubits from dephasing during
      2-qubit gate routing delays. Small extra cost per shot; useful for
      deeper-than-a-few-layers circuits on heavy-hex topologies.

    The applied options are echoed back in the summary under
    ``mitigation`` so callers (and MLflow) can record what ran.
    """
    from qiskit_ibm_runtime import (  # pyright: ignore[reportMissingImports]
        QiskitRuntimeService,
        SamplerV2,
    )

    if service is None:
        service = QiskitRuntimeService()
    assert service is not None
    if backend is None:
        backend = service.least_busy(
            operational=True,
            simulator=False,
            min_num_qubits=min_qubits,
        )

    # Runtime requires ISA-compatible circuits on hardware targets.
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_pubs: list[SamplerPub] = []
    for pub in pubs:
        try:
            qc_isa = pm.run(pub.circuit)
            isa_pubs.append(
                SamplerPub(
                    circuit=qc_isa,
                    parameter_values=getattr(pub, "parameter_values", None),
                    shots=getattr(pub, "shots", None),
                )
            )
        except Exception:
            # Keep a permissive fallback so callers still receive the original
            # Runtime validation error if transformation cannot be applied.
            isa_pubs.append(pub)

    sampler = SamplerV2(mode=backend)
    mitigation_info: dict[str, Any] = {"enabled": bool(enable_mitigation)}
    if enable_mitigation:
        try:
            sampler.options.twirling.enable_measure = True
            sampler.options.twirling.num_randomizations = "auto"
            sampler.options.twirling.shots_per_randomization = "auto"
            mitigation_info["trex"] = True
        except Exception as exc:
            mitigation_info["trex"] = False
            mitigation_info["trex_error"] = str(exc)
        try:
            sampler.options.dynamical_decoupling.enable = True
            sampler.options.dynamical_decoupling.sequence_type = dd_sequence
            mitigation_info["dd"] = dd_sequence
        except Exception as exc:
            mitigation_info["dd"] = False
            mitigation_info["dd_error"] = str(exc)
    job = sampler.run(isa_pubs)
    result = _runtime_job_result(job)
    summary = summarize_runtime_job(job)
    summary["backend_name"] = getattr(backend, "name", str(backend))
    summary["mitigation"] = mitigation_info
    return result, summary


def estimate_assignment_matrix_from_backend(
    backend: Any,
    *,
    n_qubits: int,
    default_readout_error: float = 0.03,
) -> np.ndarray:
    """Build a symmetric assignment matrix from backend readout metadata.

    Falls back to ``default_readout_error`` when backend properties are absent.
    """

    def _try_extract_readout_error() -> float | None:
        props_getter = getattr(backend, "properties", None)
        if not callable(props_getter):
            return None
        try:
            props = props_getter()
        except Exception:
            return None
        qubits = getattr(props, "qubits", None)
        if not isinstance(qubits, list):
            return None

        per_q: list[float] = []
        for q_metrics in qubits:
            if not isinstance(q_metrics, list):
                continue
            p01: float | None = None
            p10: float | None = None
            readout_err: float | None = None
            for item in q_metrics:
                name = getattr(item, "name", None)
                val = getattr(item, "value", None)
                if not isinstance(name, str) or val is None:
                    continue
                try:
                    fval = float(val)
                except Exception:
                    continue
                if name == "readout_error":
                    readout_err = fval
                elif name == "prob_meas0_prep1":
                    p01 = fval
                elif name == "prob_meas1_prep0":
                    p10 = fval
            if p01 is not None and p10 is not None:
                per_q.append(max(0.0, 0.5 * (p01 + p10)))
            elif readout_err is not None:
                per_q.append(max(0.0, readout_err))
        if not per_q:
            return None
        return float(np.mean(per_q))

    readout_error = _try_extract_readout_error()
    if readout_error is None:
        readout_error = float(default_readout_error)
    return build_assignment_matrix_from_symmetric_readout_error(
        n_qubits=n_qubits,
        readout_error_rate=readout_error,
    )


__all__ = [
    "estimate_assignment_matrix_from_backend",
    "extract_counts_sequence_from_pub_result",
    "extract_counts_sequence_from_sampler_result",
    "run_ibm_sampler_pubs",
    "summarize_runtime_job",
]
