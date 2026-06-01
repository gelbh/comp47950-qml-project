"""Bootstrap summaries, paired tests, and learning-curve fits.

Split into three submodules so each set of helpers is short enough to keep in
view:

- :mod:`qml_project.training.stats.bootstrap` — bootstrap CIs and grouped summaries.
- :mod:`qml_project.training.stats.paired` — paired effect sizes and Wilcoxon-based
  significance tests with Bonferroni correction.
- :mod:`qml_project.training.stats.curves` — learning-curve fits (power law).

The public surface is unchanged from the original ``training/stats.py`` module.
"""

from __future__ import annotations

from .bootstrap import _grouped_bootstrap_summary, bootstrap_mean_ci
from .curves import (
    fit_power_law_learning_curve,
    learning_curve_xy_for_power_law,
    power_law_fit_from_learning_curve_dataframe,
)
from .paired import (
    _paired_wilcoxon_with_bonferroni,
    paired_cohens_d,
    paired_cross_pipeline_stat_tests,
    rank_biserial_from_deltas,
    sample_efficiency_stat_tests,
)

__all__ = [
    "bootstrap_mean_ci",
    "_grouped_bootstrap_summary",
    "paired_cohens_d",
    "rank_biserial_from_deltas",
    "_paired_wilcoxon_with_bonferroni",
    "sample_efficiency_stat_tests",
    "paired_cross_pipeline_stat_tests",
    "fit_power_law_learning_curve",
    "learning_curve_xy_for_power_law",
    "power_law_fit_from_learning_curve_dataframe",
]
