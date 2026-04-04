"""
Simulation training for the variational quantum classifier.

Implements:
  - COBYLA-based optimisation (gradient-free).
  - Progressive shot schedule: 250 → 500 → 750 shots by evaluation number.
  - Multi-seed experiments for variance analysis.
  - Optional noise model support via Qiskit Aer.

Designed to work with Qiskit ≥ 2.0 primitives (V2 sampler interface).

Submodules
----------
``qml_project.training.types``
    Type aliases and dataclasses.
``qml_project.training.evaluation``
    Shot schedule, sampler execution, training, and evaluation.
``qml_project.training.multi_seed``
    Multi-seed runs and MLflow cache.
``qml_project.training.ood_sweep``
    OOD sample-efficiency sweep.
``qml_project.training.noise_aer`` / ``noise_sweep``
    Aer noise models, readout mitigation, ZNE, and noise-design sweep.
``qml_project.training.stats``
    Bootstrap CIs, learning-curve fits, pairwise tests.
"""

from qml_project.training.types import (
    DecisionRule,
    ExperimentResult,
    LossName,
    MeasurementObservable,
    MultiSeedSummary,
    SimulatedVQCRunResult,
    TrainingHistory,
    VqcAnsatzHypothesis,
    VqcNoiseSweepRunResult,
    VqcNoiseSweepTask,
    VqcOodSweepTask,
)
from qml_project.training.stats import (
    bootstrap_mean_ci,
    fit_power_law_learning_curve,
    paired_cohens_d,
    rank_biserial_from_deltas,
    sample_efficiency_stat_tests,
)
from qml_project.training.results import (
    SimulatedVQCSweepResults,
    VqcNoiseSweepResults,
)
from qml_project.training.noise_aer import (
    build_assignment_matrix_from_symmetric_readout_error,
    create_depolarizing_noise_model,
    create_noisy_sampler,
    default_vqc_ansatz_hypotheses,
    mitigate_readout_prob_vector,
    zne_extrapolate_to_zero,
)
from qml_project.training.evaluation import (
    evaluate_circuit,
    evaluate_circuit_outputs,
    evaluate_classifier,
    evaluate_vqc_win_rate,
    shots_for_eval,
    train_classifier,
    vqc_policy,
)
from qml_project.training.multi_seed import run_multi_seed_experiment
from qml_project.training.ood_sweep import run_simulated_vqc_ood_sweep
from qml_project.training.noise_sweep import run_vqc_noise_sweep

__all__ = [
    "MeasurementObservable",
    "DecisionRule",
    "LossName",
    "TrainingHistory",
    "ExperimentResult",
    "MultiSeedSummary",
    "SimulatedVQCRunResult",
    "SimulatedVQCSweepResults",
    "VqcAnsatzHypothesis",
    "VqcNoiseSweepRunResult",
    "VqcNoiseSweepResults",
    "VqcNoiseSweepTask",
    "VqcOodSweepTask",
    "shots_for_eval",
    "evaluate_circuit",
    "evaluate_circuit_outputs",
    "vqc_policy",
    "evaluate_vqc_win_rate",
    "train_classifier",
    "evaluate_classifier",
    "run_multi_seed_experiment",
    "bootstrap_mean_ci",
    "paired_cohens_d",
    "rank_biserial_from_deltas",
    "sample_efficiency_stat_tests",
    "fit_power_law_learning_curve",
    "run_simulated_vqc_ood_sweep",
    "create_depolarizing_noise_model",
    "create_noisy_sampler",
    "default_vqc_ansatz_hypotheses",
    "build_assignment_matrix_from_symmetric_readout_error",
    "mitigate_readout_prob_vector",
    "zne_extrapolate_to_zero",
    "run_vqc_noise_sweep",
]
