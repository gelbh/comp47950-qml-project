"""Picklable VQC construction for parallel sweeps."""

from __future__ import annotations

from typing import Any, Callable

from qml_project.circuit import VariationalClassifier, build_circuit


def _make_vqc_factory(
    vc_builder: Callable[[], VariationalClassifier] | None,
    circuit_kwargs: dict[str, Any] | None,
) -> Callable[[], VariationalClassifier]:
    if circuit_kwargs is not None and vc_builder is not None:
        raise ValueError("Pass at most one of vc_builder and circuit_kwargs.")
    if circuit_kwargs is not None:

        def factory_ck() -> VariationalClassifier:
            return build_circuit(**circuit_kwargs)

        return factory_ck
    if vc_builder is None:
        raise ValueError("Either vc_builder or circuit_kwargs is required.")
    return vc_builder
