"""Pool/worker entry points and the top-level noise-sweep runner."""

from __future__ import annotations

import time
from typing import Any, Callable, Sequence

import numpy as np

from qml_project.circuit import VariationalClassifier, build_circuit
from qml_project.parallel_sweep import run_pending_grid_tasks
from qml_project.training.mlflow_helpers import set_mlflow_tracking_uri
from qml_project.training.results import VqcNoiseSweepResults
from qml_project.training.types import (
    DecisionRule,
    LossName,
    MeasurementObservable,
    VqcNoiseSweepRunResult,
    VqcNoiseSweepTask,
)
from qml_project.training.vqc_factory import _make_vqc_factory

from .mlflow_io import (
    _load_vqc_noise_sweep_from_mlflow,
    _log_vqc_noise_result_to_mlflow,
)
from .single_run import _single_noise_run

_vqc_noise_pool: dict[str, Any] = {}


def _vqc_noise_pool_init(cfg: dict[str, Any]) -> None:
    _vqc_noise_pool.clear()
    _vqc_noise_pool.update(cfg)


def _vqc_noise_worker(task: VqcNoiseSweepTask) -> VqcNoiseSweepRunResult:
    p = _vqc_noise_pool
    vc = build_circuit(**p["circuit_kwargs"])
    return _single_noise_run(
        vc=vc,
        X_train=p["X_train"],
        y_train=p["y_train"],
        X_test=p["X_test"],
        y_test=p["y_test"],
        task=task,
        max_iter=p["max_iter"],
        shot_schedule=p["shot_schedule"],
        decision_rule=p["decision_rule"],
        observable=p["observable"],
        loss_name=p["loss_name"],
        expectation_qubit=p["expectation_qubit"],
        zne_scales=p["zne_scales"],
        zne_degree=p["zne_degree"],
        single_gate_error_ratio=p["single_gate_error_ratio"],
        readout_error_rate=p["readout_error_rate"],
        backend_noise_model=p["backend_noise_model"],
        apply_readout_correction=p["apply_readout_correction"],
        apply_zne=p["apply_zne"],
        log_interval=p["log_interval"],
    )


def run_vqc_noise_sweep(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    vc_builder: Callable[[], VariationalClassifier] | None = None,
    circuit_kwargs: dict[str, Any] | None = None,
    depolarizing_rates: Sequence[float] = (0.001, 0.005, 0.01, 0.02, 0.05),
    include_backend_model: bool = False,
    backend_noise_model: Any | None = None,
    shot_budgets: Sequence[int] = (8192, 4096, 2048, 1024, 512),
    seeds: Sequence[int] = tuple(range(10)),
    max_iter: int = 200,
    shot_schedule: dict[int, int] | None = None,
    decision_rule: DecisionRule = "argmax",
    observable: MeasurementObservable = "bitstring_probs",
    loss_name: LossName = "softmax_nll",
    expectation_qubit: int = 0,
    single_gate_error_ratio: float = 0.2,
    readout_error_rate: float = 0.02,
    apply_readout_correction: bool = True,
    apply_zne: bool = True,
    zne_scales: Sequence[float] = (1.0, 2.0, 3.0),
    zne_degree: int = 1,
    mlflow_experiment: str | None = None,
    mlflow_run_prefix: str = "vqc-noise-design",
    use_cache: bool = True,
    verbose: bool = True,
    max_workers: int | None = None,
    log_interval: int = 20,
) -> VqcNoiseSweepResults:
    """Sweep simulated VQC over noise levels, shot budgets, and seeds.

    This helper is designed for §5.6-style analysis:
      - 5-level depolarising noise sweep
      - optional backend-specific noise profile
      - readout correction and ZNE mitigation variants
      - shot-budget sensitivity (8192 → 512)

    When ``mlflow_experiment`` is set, each grid point is logged to MLflow as
    soon as it finishes (parent process), so a later run with ``use_cache=True``
    can skip points already stored under ``mlflow_run_prefix``.
    """
    sweep_started = time.perf_counter()
    if (max_workers is not None and max_workers > 1) and circuit_kwargs is None:
        raise ValueError("circuit_kwargs is required when max_workers > 1.")

    try:
        import qiskit_aer  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "run_vqc_noise_sweep requires the `qiskit-aer` package (Aer noise "
            "models and SamplerV2). Install it in the same environment as your "
            "Jupyter kernel, e.g. `make env-qiskit` or "
            "`UV_PROJECT_ENVIRONMENT=.venv-qiskit uv sync --group qiskit`, then "
            "select the `.venv-qiskit` kernel."
        ) from exc

    factory = _make_vqc_factory(vc_builder, circuit_kwargs)
    probe_vc = factory()

    mlflow_available = False
    if mlflow_experiment:
        try:
            import mlflow

            set_mlflow_tracking_uri()
            mlflow.set_experiment(mlflow_experiment)
            mlflow_available = True
        except ImportError:
            mlflow_available = False
            if verbose:
                print("Warning: MLflow not available; skipping noise-sweep logging.")

    task_grid: list[VqcNoiseSweepTask] = []
    for seed in seeds:
        for shots in shot_budgets:
            for rate in depolarizing_rates:
                task_grid.append(
                    VqcNoiseSweepTask(
                        noise_profile="depolarizing",
                        noise_level=float(rate),
                        shots=int(shots),
                        seed=int(seed),
                    )
                )
            if include_backend_model:
                task_grid.append(
                    VqcNoiseSweepTask(
                        noise_profile="backend",
                        noise_level=None,
                        shots=int(shots),
                        seed=int(seed),
                    )
                )

    cache: dict[tuple[str, float | None, int, int], VqcNoiseSweepRunResult] = {}
    if use_cache and mlflow_available and mlflow_experiment:
        cache = _load_vqc_noise_sweep_from_mlflow(
            mlflow_experiment,
            mlflow_run_prefix,
            probe_ansatz=str(probe_vc.ansatz),
        )

    ordered: list[VqcNoiseSweepRunResult | None] = []
    pending: list[tuple[int, VqcNoiseSweepTask]] = []
    for i, task in enumerate(task_grid):
        key = (task.noise_profile, task.noise_level, task.shots, task.seed)
        if key in cache:
            ordered.append(cache[key])
            if verbose and (max_workers is None or max_workers <= 1):
                print(
                    f"[noise {i + 1}/{len(task_grid)}] "
                    f"profile={task.noise_profile} level={task.noise_level} "
                    f"shots={task.shots} seed={task.seed} (cached)"
                )
        else:
            ordered.append(None)
            pending.append((i, task))

    def _log_noise(r: VqcNoiseSweepRunResult) -> None:
        if not mlflow_available:
            return
        _log_vqc_noise_result_to_mlflow(
            r,
            mlflow_run_prefix=mlflow_run_prefix,
            probe_vc=probe_vc,
            max_iter=max_iter,
            decision_rule=decision_rule,
            observable=observable,
            loss_name=loss_name,
            expectation_qubit=expectation_qubit,
        )

    if pending:

        def _run_serial(task: VqcNoiseSweepTask) -> VqcNoiseSweepRunResult:
            return _single_noise_run(
                vc=factory(),
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                task=task,
                max_iter=max_iter,
                shot_schedule=shot_schedule,
                decision_rule=decision_rule,
                observable=observable,
                loss_name=loss_name,
                expectation_qubit=expectation_qubit,
                zne_scales=zne_scales,
                zne_degree=zne_degree,
                single_gate_error_ratio=single_gate_error_ratio,
                readout_error_rate=readout_error_rate,
                backend_noise_model=backend_noise_model,
                apply_readout_correction=apply_readout_correction,
                apply_zne=apply_zne,
                log_interval=log_interval,
            )

        def _after_task(_j: int, result: VqcNoiseSweepRunResult) -> None:
            _log_noise(result)

        cfg: dict[str, Any] | None = None
        if max_workers is not None and max_workers > 1:
            assert circuit_kwargs is not None
            cfg = {
                "circuit_kwargs": dict(circuit_kwargs),
                "X_train": X_train,
                "y_train": y_train,
                "X_test": X_test,
                "y_test": y_test,
                "max_iter": max_iter,
                "shot_schedule": shot_schedule,
                "decision_rule": decision_rule,
                "observable": observable,
                "loss_name": loss_name,
                "expectation_qubit": expectation_qubit,
                "zne_scales": tuple(float(s) for s in zne_scales),
                "zne_degree": int(zne_degree),
                "single_gate_error_ratio": float(single_gate_error_ratio),
                "readout_error_rate": float(readout_error_rate),
                "backend_noise_model": backend_noise_model,
                "apply_readout_correction": bool(apply_readout_correction),
                "apply_zne": bool(apply_zne),
                "log_interval": int(log_interval),
            }

        run_pending_grid_tasks(
            pending,
            ordered,
            run_serial=_run_serial,
            parallel_worker=_vqc_noise_worker,
            max_workers=max_workers,
            tqdm_desc=mlflow_run_prefix,
            initializer=_vqc_noise_pool_init if cfg is not None else None,
            initargs=(cfg,) if cfg is not None else (),
            on_task_complete=_after_task,
        )

    total_tasks = len(task_grid)
    return VqcNoiseSweepResults(
        results=[r for r in ordered if r is not None],
        sweep_metadata={
            "elapsed_wall_time_s": float(time.perf_counter() - sweep_started),
            "max_workers": (
                int(max_workers)
                if (max_workers is not None and max_workers > 1)
                else 1
            ),
            "n_tasks": int(total_tasks),
            "n_cached": int(total_tasks - len(pending)),
        },
    )


__all__ = ["run_vqc_noise_sweep"]
