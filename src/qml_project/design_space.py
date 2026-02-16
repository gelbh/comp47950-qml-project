"""
Design-space exploration for the [20]-style variational quantum classifier.

Systematically varies circuit hyperparameters (qubit count, depth, CZ strategy,
feature count) and aggregates multi-seed results for comparison and device
circuit selection.

Addresses [20] Section 6 discussion points:
  - Variance across CZ configurations.
  - Width–depth trade-off (inflection point).
  - Circuit selection for NISQ device inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from qml_project.circuit import CZStrategy, VariationalClassifier, build_circuit
from qml_project.training import MultiSeedSummary, run_multi_seed_experiment

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class CircuitConfig:
    """A single point in the circuit design space."""

    name: str
    n_qubits: int
    n_features: int
    n_classes: int
    n_layers: int | None = None
    cz_strategy: CZStrategy = "linear"
    cz_seed: int = 42

    @property
    def label(self) -> str:
        """Short label for plots/tables."""
        layers_str = f"L{self.n_layers}" if self.n_layers else "Lauto"
        return f"Q{self.n_qubits}_{layers_str}_{self.cz_strategy}"

    def build(self) -> VariationalClassifier:
        """Build a fresh ``VariationalClassifier`` from this config."""
        kwargs: dict[str, Any] = dict(
            n_qubits=self.n_qubits,
            n_features=self.n_features,
            n_classes=self.n_classes,
            cz_strategy=self.cz_strategy,
            cz_seed=self.cz_seed,
        )
        if self.n_layers is not None:
            kwargs["n_layers"] = self.n_layers
        return build_circuit(**kwargs)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class DesignSpaceResult:
    """Result from one configuration in the design space."""

    config: CircuitConfig
    summary: MultiSeedSummary
    circuit_depth: int
    n_trainable: int
    n_cz_gates: int
    n_total_gates: int
    n_layers_actual: int = 0


# ---------------------------------------------------------------------------
# Grid runner
# ---------------------------------------------------------------------------


def run_design_space(
    configs: list[CircuitConfig],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    seeds: list[int] | None = None,
    n_seeds: int = 5,
    max_iter: int = 100,
    test_shots: int = 300,
    sampler_factory: Callable[[int], Any] | None = None,
    verbose: bool = True,
    log_interval: int = 25,
) -> list[DesignSpaceResult]:
    """
    Run multi-seed experiments across a list of circuit configurations.

    Parameters
    ----------
    configs : list[CircuitConfig]
        Design-space points to evaluate.
    X_train, y_train, X_test, y_test : ndarray
        Preprocessed (angle-mapped) data.
    seeds : list[int] or None
        Explicit seed list.  Overrides *n_seeds* if given.
    n_seeds : int
        Number of random seeds per config (default 5).
    max_iter : int
        COBYLA iterations per seed.
    test_shots : int
        Shots for test-set evaluation.
    sampler_factory : callable or None
        ``seed -> sampler``.  Pass for noisy experiments.
    verbose : bool
        Print progress banners.
    log_interval : int
        How often to log within each training run.

    Returns
    -------
    list[DesignSpaceResult]
        One entry per configuration.
    """
    results: list[DesignSpaceResult] = []

    for i, cfg in enumerate(configs):
        if verbose:
            print(f"\n{'#' * 70}")
            print(f"Config {i + 1}/{len(configs)}: {cfg.name} ({cfg.label})")
            print(
                f"  Qubits={cfg.n_qubits}, Features={cfg.n_features}, "
                f"Layers={cfg.n_layers or 'auto'}, CZ={cfg.cz_strategy}"
            )
            print(f"{'#' * 70}")

        vc_sample = cfg.build()
        ops = vc_sample.circuit.count_ops()
        ops_by_name = {getattr(k, "name", str(k)): v for k, v in ops.items()}

        summary = run_multi_seed_experiment(
            vc_builder=cfg.build,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            seeds=seeds,
            n_seeds=n_seeds,
            max_iter=max_iter,
            test_shots=test_shots,
            sampler_factory=sampler_factory,
            verbose=verbose,
            log_interval=log_interval,
        )

        results.append(
            DesignSpaceResult(
                config=cfg,
                summary=summary,
                circuit_depth=vc_sample.circuit.depth(),
                n_trainable=vc_sample.n_trainable,
                n_cz_gates=ops_by_name.get("cz", 0),
                n_total_gates=sum(
                    v for k, v in ops_by_name.items() if k != "barrier"
                ),
                n_layers_actual=vc_sample.n_layers,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Summarisation
# ---------------------------------------------------------------------------


def summarize_results(results: list[DesignSpaceResult]) -> pd.DataFrame:
    """
    Create a summary DataFrame from design-space results.

    Returns one row per configuration with circuit metadata and accuracy
    statistics.
    """
    rows = []
    for r in results:
        rows.append(
            {
                "Config": r.config.name,
                "Qubits": r.config.n_qubits,
                "Features": r.config.n_features,
                "Layers": r.n_layers_actual,
                "CZ strategy": r.config.cz_strategy,
                "Depth": r.circuit_depth,
                "Trainable": r.n_trainable,
                "CZ gates": r.n_cz_gates,
                "Total gates": r.n_total_gates,
                "Acc (mean)": round(r.summary.test_accuracy_mean, 4),
                "Acc (std)": round(r.summary.test_accuracy_std, 4),
                "Acc (min)": round(r.summary.test_accuracy_min, 4),
                "Acc (max)": round(r.summary.test_accuracy_max, 4),
                "Train time (s)": round(r.summary.training_time_mean, 1),
                "Infer time (s)": round(r.summary.inference_time_mean, 3),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Device circuit selection
# ---------------------------------------------------------------------------


def select_device_circuits(
    results: list[DesignSpaceResult],
    *,
    max_depth: int = 50,
    max_cz: int = 30,
    min_accuracy: float = 0.0,
    top_k: int = 3,
) -> list[DesignSpaceResult]:
    """
    Select the best configurations for real-device inference.

    Filters by hardware-friendliness (circuit depth, CZ gate count) and
    minimum accuracy threshold, then returns the *top_k* by mean test
    accuracy.

    Parameters
    ----------
    results : list[DesignSpaceResult]
        Full design-space results.
    max_depth : int
        Maximum transpiled circuit depth (pre-transpilation proxy).
    max_cz : int
        Maximum number of CZ gates.
    min_accuracy : float
        Minimum mean test accuracy to be considered.
    top_k : int
        Number of top candidates to return.

    Returns
    -------
    list[DesignSpaceResult]
        Up to *top_k* configurations, sorted by accuracy (descending).
    """
    candidates = [
        r
        for r in results
        if r.circuit_depth <= max_depth
        and r.n_cz_gates <= max_cz
        and r.summary.test_accuracy_mean >= min_accuracy
    ]
    candidates.sort(
        key=lambda r: (-r.summary.test_accuracy_mean, r.circuit_depth)
    )
    return candidates[:top_k]


# ---------------------------------------------------------------------------
# Preset configuration generators
# ---------------------------------------------------------------------------


def qubit_sweep_configs(
    qubit_range: list[int],
    n_features: int,
    n_classes: int,
    *,
    n_layers: int | None = None,
    cz_strategy: CZStrategy = "linear",
    dataset_name: str = "",
) -> list[CircuitConfig]:
    """Generate configs varying qubit count with other parameters fixed."""
    return [
        CircuitConfig(
            name=f"{dataset_name} Q={q}" if dataset_name else f"Q={q}",
            n_qubits=q,
            n_features=n_features,
            n_classes=n_classes,
            n_layers=n_layers,
            cz_strategy=cz_strategy,
        )
        for q in qubit_range
    ]


def depth_sweep_configs(
    layer_range: list[int],
    n_qubits: int,
    n_features: int,
    n_classes: int,
    *,
    cz_strategy: CZStrategy = "linear",
    dataset_name: str = "",
) -> list[CircuitConfig]:
    """Generate configs varying layer count with other parameters fixed."""
    return [
        CircuitConfig(
            name=f"{dataset_name} L={l}" if dataset_name else f"L={l}",
            n_qubits=n_qubits,
            n_features=n_features,
            n_classes=n_classes,
            n_layers=l,
            cz_strategy=cz_strategy,
        )
        for l in layer_range
    ]


def cz_sweep_configs(
    strategies: list[CZStrategy],
    n_qubits: int,
    n_features: int,
    n_classes: int,
    *,
    n_layers: int | None = None,
    cz_seeds: list[int] | None = None,
    dataset_name: str = "",
) -> list[CircuitConfig]:
    """
    Generate configs varying CZ strategy.

    For ``"random"`` strategy, multiple *cz_seeds* can be provided to
    capture CZ-configuration variance ([20] Section 6).
    """
    configs: list[CircuitConfig] = []
    for strat in strategies:
        if strat == "random" and cz_seeds is not None:
            for cs in cz_seeds:
                configs.append(
                    CircuitConfig(
                        name=(
                            f"{dataset_name} CZ={strat} (seed={cs})"
                            if dataset_name
                            else f"CZ={strat} (seed={cs})"
                        ),
                        n_qubits=n_qubits,
                        n_features=n_features,
                        n_classes=n_classes,
                        n_layers=n_layers,
                        cz_strategy=strat,
                        cz_seed=cs,
                    )
                )
        else:
            configs.append(
                CircuitConfig(
                    name=(
                        f"{dataset_name} CZ={strat}"
                        if dataset_name
                        else f"CZ={strat}"
                    ),
                    n_qubits=n_qubits,
                    n_features=n_features,
                    n_classes=n_classes,
                    n_layers=n_layers,
                    cz_strategy=strat,
                )
            )
    return configs
