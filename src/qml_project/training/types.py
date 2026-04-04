"""Type aliases and dataclasses for VQC training pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

MeasurementObservable = Literal["bitstring_probs", "z_expectation"]
DecisionRule = Literal["argmax", "expectation_threshold"]
LossName = Literal[
    "softmax_nll",
    "cross_entropy_expectation",
    "hinge_expectation",
]


@dataclass
class TrainingHistory:
    """Records training progress across optimiser evaluations."""

    losses: list[float] = field(default_factory=list)
    train_accuracies: list[float] = field(default_factory=list)
    test_accuracies: list[float] = field(default_factory=list)
    eval_numbers: list[int] = field(default_factory=list)
    shot_counts: list[int] = field(default_factory=list)
    best_weights: np.ndarray | None = None
    best_loss: float = float("inf")
    total_training_time: float = 0.0
    total_evals: int = 0


@dataclass
class ExperimentResult:
    """Result from a single training run (one seed)."""

    seed: int
    best_weights: np.ndarray
    history: TrainingHistory
    test_accuracy: float
    test_predictions: np.ndarray
    test_class_probs: np.ndarray
    training_time: float
    inference_time: float
    balanced_accuracy: float | None = None
    mcc: float | None = None


@dataclass
class MultiSeedSummary:
    """Aggregated results across multiple random seeds."""

    per_seed: list[ExperimentResult]
    test_accuracy_mean: float
    test_accuracy_std: float
    test_accuracy_min: float
    test_accuracy_max: float
    training_time_mean: float
    inference_time_mean: float
    n_seeds: int


@dataclass
class SimulatedVQCRunResult:
    """Single simulated VQC run in the OOD sample-efficiency sweep."""

    train_size: int
    seed: int
    test_accuracy: float
    balanced_accuracy: float
    mcc: float
    win_rate: float | None
    training_time: float
    inference_time: float
    final_loss: float
    ansatz: str
    observable: MeasurementObservable
    decision_rule: DecisionRule
    loss_name: LossName


@dataclass(frozen=True)
class VqcOodSweepTask:
    """One simulated VQC OOD run (encoded train subset + metadata)."""

    subset_X: np.ndarray
    subset_y: np.ndarray
    seed: int
    train_size: int


@dataclass(frozen=True)
class VqcAnsatzHypothesis:
    """Design-time hypothesis for a concrete ansatz choice."""

    ansatz: str
    hypothesis: str
    expected_strength: str
    primary_risk: str


@dataclass
class VqcNoiseSweepRunResult:
    """One VQC run for a noise profile x shot budget x seed."""

    noise_profile: str
    noise_level: float | None
    shots: int
    seed: int
    ansatz: str
    training_time: float
    inference_time: float
    final_loss: float
    test_accuracy_raw: float
    balanced_accuracy_raw: float
    mcc_raw: float
    test_accuracy_readout: float | None = None
    balanced_accuracy_readout: float | None = None
    mcc_readout: float | None = None
    test_accuracy_zne: float | None = None
    balanced_accuracy_zne: float | None = None
    mcc_zne: float | None = None
    test_accuracy_readout_zne: float | None = None
    balanced_accuracy_readout_zne: float | None = None
    mcc_readout_zne: float | None = None


@dataclass(frozen=True)
class VqcNoiseSweepTask:
    """One task in the VQC noise-design sweep."""

    noise_profile: str
    noise_level: float | None
    shots: int
    seed: int
