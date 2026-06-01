"""Canonical MLflow experiment namespace and standard run tags.

The namespace is dot-separated (``nim.<pipeline>.<stage>``) so self-hosted
MLflow renders runs in alphabetical, grouped order. Helpers here return
experiment names and tag dictionaries, keeping the convention in one place so
notebook parts never construct experiment names ad-hoc.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping

Pipeline = Literal["classical", "vqc", "qsvm", "selection", "device", "final"]
Stage = Literal[
    "baseline",
    "ablation",
    "kernel_aligned",
    "pilot",
    "tuning",
    "robustness",
    "ood",
    "selection",
    "inference",
    "final",
]

NAMESPACE_ROOT = "nim"


def resolve_experiment(
    pipeline: Pipeline,
    stage: Stage,
    *,
    variant: str | None = None,
    root: str = NAMESPACE_ROOT,
) -> str:
    """Return the canonical experiment name for a (pipeline, stage) pair.

    An optional ``variant`` appends a fourth level (e.g. a sub-study tag).
    """
    parts: list[str] = [root, pipeline, stage]
    if variant:
        parts.append(variant)
    return ".".join(parts)


def standard_tags(
    *,
    pipeline: Pipeline,
    stage: Stage,
    encoding: str | None = None,
    symmetry: str | None = None,
    train_size: int | str | None = None,
    seed: int | None = None,
    ansatz: str | None = None,
    loss_name: str | None = None,
    kernel_backend: str | None = None,
    c_svc: float | None = None,
    plan_todo: str | None = None,
    include_nim_sum: bool | None = None,
    variant_id: str | None = None,
    encoding_cache_revision: str | None = None,
    config_id: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Return MLflow tag dict with the project's standard keys populated."""
    tags: dict[str, str] = {
        "pipeline": str(pipeline),
        "stage": str(stage),
    }
    if encoding is not None:
        tags["encoding"] = str(encoding)
    if symmetry is not None:
        tags["symmetry"] = str(symmetry)
    if train_size is not None:
        tags["train_size"] = str(train_size)
    if seed is not None:
        tags["seed"] = str(seed)
    if ansatz is not None:
        tags["ansatz"] = str(ansatz)
    if loss_name is not None:
        tags["loss_name"] = str(loss_name)
    if kernel_backend is not None:
        tags["kernel_backend"] = str(kernel_backend)
    if c_svc is not None:
        tags["C"] = str(c_svc)
    if plan_todo is not None:
        tags["plan_todo"] = str(plan_todo)
    if include_nim_sum is not None:
        tags["include_nim_sum"] = "true" if include_nim_sum else "false"
    if variant_id:
        tags["variant_id"] = str(variant_id)
    if encoding_cache_revision:
        tags["encoding_cache_revision"] = str(encoding_cache_revision)
    if config_id:
        tags["config_id"] = str(config_id)
    if extra:
        for k, v in extra.items():
            tags[str(k)] = str(v)
    return tags


def run_name(
    stage: str,
    config_signature: str,
    *,
    train_size: int | str | None = None,
    seed: int | None = None,
) -> str:
    """Return a canonical MLflow run name: ``stage|sig|n=X|s=Y``."""
    parts: list[str] = [stage, config_signature]
    if train_size is not None:
        parts.append(f"n={train_size}")
    if seed is not None:
        parts.append(f"s={seed}")
    return "|".join(parts)


CANONICAL_EXPERIMENTS: dict[str, str] = {
    "classical_baseline": resolve_experiment("classical", "baseline"),
    "classical_ablation": resolve_experiment("classical", "ablation"),
    "classical_kernel_aligned": resolve_experiment("classical", "kernel_aligned"),
    "vqc_pilot": resolve_experiment("vqc", "pilot"),
    "vqc_architecture_diagnostics": resolve_experiment(
        "vqc", "pilot", variant="architecture-diagnostics"
    ),
    "vqc_tuning": resolve_experiment("vqc", "tuning"),
    "vqc_robustness": resolve_experiment("vqc", "robustness"),
    "vqc_ood": resolve_experiment("vqc", "ood"),
    "qsvm_tuning": resolve_experiment("qsvm", "tuning"),
    "selection_quantum": resolve_experiment("selection", "selection"),
    "device_inference": resolve_experiment("device", "inference"),
    "final_three_way": resolve_experiment("final", "final"),
}


__all__ = [
    "Pipeline",
    "Stage",
    "NAMESPACE_ROOT",
    "CANONICAL_EXPERIMENTS",
    "resolve_experiment",
    "standard_tags",
    "run_name",
]
