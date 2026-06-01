"""``Winner`` dataclass for quantum selection in ``notebooks/qml_project``.

Section 07 constructs ``Winner`` instances via
``qml_project.pareto_selection.build_pareto_quantum_selection`` for §8–§11.
``Winner`` lives here (not in a notebook cell) so optional pickles under
``notebooks/.workflow_cache/`` can be unpickled when re-running device cells.

The dataclass is intentionally lightweight — it carries only the minimum
state needed to reconstruct a deploy-time configuration and to filter the
upstream workflow frame back to the winning rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class Winner:
    """Selected quantum configuration produced by Section 07.

    Attributes
    ----------
    pipeline
        ``"vqc"`` or ``"qsvm"``.
    config_id
        Synthetic ``selection_id`` built by joining the grouping columns
        with ``"|"`` (e.g. ``"sv|C=10|sym=canonical|amplitude"``). For
        VQC this is the same string as the upstream ``config_id``
        column; for QSVM it embeds the encoding.
    encoding
        Encoding name (``"amplitude"``, ``"angle"``, …) or ``None`` for
        VQC when the column is missing.
    mean_accuracy, std_accuracy
        Aggregate balanced accuracy across seeds at the deploy-time
        ``train_size``.
    mean_cost
        Mean ``training_time_s`` at the deploy-time ``train_size``, or
        ``None`` when the cost column is absent.
    train_size_used
        Training-set size at which the aggregate was computed (usually
        ``max(train_size)``).
    rationale
        Human-readable description of why this row was picked.
    match_keys
        Column-value pairs used to slice the upstream ``workflow_df``
        back to this winner (e.g. ``{"variant_id": "...", "encoding":
        "amplitude"}``). :func:`qml_project.pareto_selection.filter_workflow_rows_to_winner`
        consumes these when building ``quantum_winner_rows_by_pipeline``.
    row
        The full ``selection_table`` row as a dict, for MLflow logging
        and audit.
    """

    pipeline: str
    config_id: str
    encoding: str | None
    mean_accuracy: float
    std_accuracy: float
    mean_cost: float | None
    train_size_used: int | None
    rationale: str
    match_keys: Mapping[str, Any] = field(default_factory=dict)
    row: Mapping[str, Any] = field(default_factory=dict)


__all__ = ["Winner"]
