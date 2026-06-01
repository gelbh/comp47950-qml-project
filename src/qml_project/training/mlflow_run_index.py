"""Thin typed wrapper around MLflow ``Run`` objects for cache loaders.

:P:class:`RunRow` mirrors the fields loaders actually read from
``MlflowClient.search_runs`` results. Use :func:`run_row_from_mlflow` inside
pipeline-specific filters instead of repeating attribute paths.

This module does **not** replace pipeline-specific grid logic — only shared parsing.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RunRow(BaseModel):
    """Subset of one MLflow run row used by sweep resume code."""

    model_config = {"frozen": True}

    run_id: str
    status: str
    run_name: str | None = None
    params: dict[str, str]
    metrics: dict[str, float]
    tags: dict[str, str] = Field(default_factory=dict)


def run_row_from_mlflow(run: Any) -> RunRow:
    """Build a :class:`RunRow` from an ``mlflow.entities.Run``."""
    info = run.info
    data = run.data
    tags_raw = getattr(data, "tags", None) or {}
    tags = {str(k): str(v) for k, v in tags_raw.items()}
    name = getattr(info, "run_name", None)
    if not name:
        name = tags.get("mlflow.runName")
    return RunRow(
        run_id=str(info.run_id),
        status=str(info.status),
        run_name=str(name) if name else None,
        params=dict(data.params),
        metrics=dict(data.metrics),
        tags=tags,
    )


def iter_finished_run_rows(
    client: Any,
    experiment_id: str,
    *,
    order_by: list[str] | None = None,
    max_results: int = 10_000,
    filter_string: str | None = None,
):
    """Yield :class:`RunRow` for each run returned by ``search_runs``."""
    kwargs: dict[str, Any] = {
        "experiment_ids": [experiment_id],
        "max_results": int(max_results),
    }
    if order_by is not None:
        kwargs["order_by"] = order_by
    if filter_string is not None:
        kwargs["filter_string"] = filter_string
    for run in client.search_runs(**kwargs):
        yield run_row_from_mlflow(run)


__all__ = ["RunRow", "iter_finished_run_rows", "run_row_from_mlflow"]
