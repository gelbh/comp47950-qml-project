"""VQC noise-design sweep (depolarising, readout correction, ZNE)."""

from __future__ import annotations

import time
from typing import Any, Callable, Sequence

import numpy as np

from qml_project.circuit import VariationalClassifier, build_circuit
from qml_project.parallel_sweep import map_parallel_or_serial
from qml_project.training.evaluation import (
    evaluate_circuit_outputs,
    train_classifier,
    _predict_from_outputs,
)
from qml_project.training.metrics import _metrics_from_preds
from qml_project.training.mlflow_helpers import _set_mlflow_tracking_uri
from qml_project.training.noise_aer import (
    build_assignment_matrix_from_symmetric_readout_error,
    create_depolarizing_noise_model,
    create_noisy_sampler,
    _zne_extrapolate_outputs,
)
from qml_project.training.results import VqcNoiseSweepResults
from qml_project.training.types import (
    DecisionRule,
    LossName,
    MeasurementObservable,
    VqcNoiseSweepRunResult,
    VqcNoiseSweepTask,
)
from qml_project.training.vqc_factory import _make_vqc_factory

_vqc_noise_pool: dict[str, Any] = {}


def _vqc_noise_pool_init(cfg: dict[str, Any]) -> None:
    _vqc_noise_pool.clear()
    _vqc_noise_pool.update(cfg)


def _single_noise_run(
    *,
    vc: VariationalClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task: VqcNoiseSweepTask,
    max_iter: int,
    shot_schedule: dict[int, int] | None,
    decision_rule: DecisionRule,
    observable: MeasurementObservable,
    loss_name: LossName,
    expectation_qubit: int,
    zne_scales: Sequence[float],
    zne_degree: int,
    single_gate_error_ratio: float,
    readout_error_rate: float,
    backend_noise_model: Any | None,
    apply_readout_correction: bool,
    apply_zne: bool,
    log_interval: int,
) -> VqcNoiseSweepRunResult:
    """Train/evaluate one point of the noise sweep."""
    seed = int(task.seed)
    if task.noise_profile == "depolarizing":
        base_rate = float(task.noise_level or 0.0)
        noise_model = create_depolarizing_noise_model(
            cz_error_rate=base_rate,
            single_gate_error_rate=single_gate_error_ratio * base_rate,
            readout_error_rate=readout_error_rate,
        )
    elif task.noise_profile == "backend":
        if backend_noise_model is None:
            raise ValueError("backend_noise_model is required for backend profile.")
        noise_model = backend_noise_model
    else:
        raise ValueError(f"Unsupported noise profile: {task.noise_profile}")

    sampler = create_noisy_sampler(noise_model, seed=seed)
    best_weights, history = train_classifier(
        vc,
        X_train,
        y_train,
        X_test,
        y_test,
        max_iter=max_iter,
        shot_schedule=shot_schedule,
        seed=seed,
        test_shots=int(task.shots),
        sampler=sampler,
        observable=observable,
        decision_rule=decision_rule,
        loss_name=loss_name,
        expectation_qubit=expectation_qubit,
        verbose=False,
        log_interval=log_interval,
        mlflow_experiment=None,
    )

    t0 = time.perf_counter()
    base_outputs = evaluate_circuit_outputs(
        vc,
        X_test,
        best_weights,
        int(task.shots),
        sampler,
        expectation_qubit=expectation_qubit,
    )
    inference_time = time.perf_counter() - t0
    raw_preds = _predict_from_outputs(base_outputs, decision_rule=decision_rule)
    acc_raw, bal_raw, mcc_raw = _metrics_from_preds(y_test, raw_preds)

    readout_matrix: np.ndarray | None = None
    if apply_readout_correction:
        if task.noise_profile == "depolarizing" and readout_error_rate > 0:
            readout_matrix = build_assignment_matrix_from_symmetric_readout_error(
                n_qubits=vc.n_qubits,
                readout_error_rate=readout_error_rate,
            )

    readout_metrics: tuple[float, float, float] | None = None
    if readout_matrix is not None:
        readout_outputs = evaluate_circuit_outputs(
            vc,
            X_test,
            best_weights,
            int(task.shots),
            sampler,
            expectation_qubit=expectation_qubit,
            readout_assignment_matrix=readout_matrix,
        )
        readout_preds = _predict_from_outputs(readout_outputs, decision_rule=decision_rule)
        readout_metrics = _metrics_from_preds(y_test, readout_preds)

    zne_metrics: tuple[float, float, float] | None = None
    readout_zne_metrics: tuple[float, float, float] | None = None
    if apply_zne and task.noise_profile == "depolarizing":
        outputs_by_scale: list[dict[str, np.ndarray]] = []
        outputs_by_scale_readout: list[dict[str, np.ndarray]] = []
        for scale in zne_scales:
            scaled_rate = min(float(task.noise_level or 0.0) * float(scale), 0.49)
            scaled_noise = create_depolarizing_noise_model(
                cz_error_rate=scaled_rate,
                single_gate_error_rate=single_gate_error_ratio * scaled_rate,
                readout_error_rate=readout_error_rate,
            )
            scaled_sampler = create_noisy_sampler(scaled_noise, seed=seed)
            outputs_by_scale.append(
                evaluate_circuit_outputs(
                    vc,
                    X_test,
                    best_weights,
                    int(task.shots),
                    scaled_sampler,
                    expectation_qubit=expectation_qubit,
                )
            )
            if readout_matrix is not None:
                outputs_by_scale_readout.append(
                    evaluate_circuit_outputs(
                        vc,
                        X_test,
                        best_weights,
                        int(task.shots),
                        scaled_sampler,
                        expectation_qubit=expectation_qubit,
                        readout_assignment_matrix=readout_matrix,
                    )
                )

        zne_outputs = _zne_extrapolate_outputs(
            outputs_by_scale,
            scales=zne_scales,
            degree=zne_degree,
        )
        zne_preds = _predict_from_outputs(zne_outputs, decision_rule=decision_rule)
        zne_metrics = _metrics_from_preds(y_test, zne_preds)

        if outputs_by_scale_readout:
            zne_outputs_readout = _zne_extrapolate_outputs(
                outputs_by_scale_readout,
                scales=zne_scales,
                degree=zne_degree,
            )
            zne_preds_readout = _predict_from_outputs(
                zne_outputs_readout, decision_rule=decision_rule
            )
            readout_zne_metrics = _metrics_from_preds(y_test, zne_preds_readout)

    return VqcNoiseSweepRunResult(
        noise_profile=task.noise_profile,
        noise_level=task.noise_level,
        shots=int(task.shots),
        seed=seed,
        ansatz=str(vc.ansatz),
        training_time=float(history.total_training_time),
        inference_time=float(inference_time),
        final_loss=float(history.best_loss),
        test_accuracy_raw=acc_raw,
        balanced_accuracy_raw=bal_raw,
        mcc_raw=mcc_raw,
        test_accuracy_readout=None if readout_metrics is None else readout_metrics[0],
        balanced_accuracy_readout=None if readout_metrics is None else readout_metrics[1],
        mcc_readout=None if readout_metrics is None else readout_metrics[2],
        test_accuracy_zne=None if zne_metrics is None else zne_metrics[0],
        balanced_accuracy_zne=None if zne_metrics is None else zne_metrics[1],
        mcc_zne=None if zne_metrics is None else zne_metrics[2],
        test_accuracy_readout_zne=None
        if readout_zne_metrics is None
        else readout_zne_metrics[0],
        balanced_accuracy_readout_zne=None
        if readout_zne_metrics is None
        else readout_zne_metrics[1],
        mcc_readout_zne=None if readout_zne_metrics is None else readout_zne_metrics[2],
    )


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
    force_rerun: bool = False,
    verbose: bool = True,
    max_workers: int | None = None,
    use_tqdm: bool = True,
    log_interval: int = 20,
) -> VqcNoiseSweepResults:
    """
    Sweep simulated VQC over noise levels, shot budgets, and seeds.

    This helper is designed for §5.6-style analysis:
      - 5-level depolarising noise sweep
      - optional backend-specific noise profile
      - readout correction and ZNE mitigation variants
      - shot-budget sensitivity (8192 → 512)
    """
    if (max_workers is not None and max_workers > 1) and circuit_kwargs is None:
        raise ValueError("circuit_kwargs is required when max_workers > 1.")

    factory = _make_vqc_factory(vc_builder, circuit_kwargs)
    probe_vc = factory()

    mlflow_available = False
    if mlflow_experiment:
        try:
            import mlflow

            _set_mlflow_tracking_uri()
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
    if (
        use_cache
        and not force_rerun
        and mlflow_available
        and mlflow_experiment
    ):
        try:
            from mlflow.tracking import MlflowClient

            client = MlflowClient()
            exp = client.get_experiment_by_name(mlflow_experiment)
            if exp is not None:
                runs = client.search_runs(
                    experiment_ids=[exp.experiment_id],
                    order_by=["end_time DESC"],
                    max_results=20_000,
                )
                for run in runs:
                    if run.info.status != "FINISHED":
                        continue
                    p = run.data.params
                    m = run.data.metrics
                    if p.get("pipeline") != "simulated_vqc_noise":
                        continue
                    if p.get("run_prefix") != mlflow_run_prefix:
                        continue
                    if p.get("ansatz") != str(probe_vc.ansatz):
                        continue
                    try:
                        prof = p["noise_profile"]
                        level = (
                            None if p.get("noise_level") in (None, "none") else float(p["noise_level"])
                        )
                        shots = int(p["shots"])
                        seed = int(p["seed"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    key = (prof, level, shots, seed)
                    if key in cache:
                        continue
                    cache[key] = VqcNoiseSweepRunResult(
                        noise_profile=prof,
                        noise_level=level,
                        shots=shots,
                        seed=seed,
                        ansatz=str(probe_vc.ansatz),
                        training_time=float(m.get("training_time", 0.0)),
                        inference_time=float(m.get("inference_time", 0.0)),
                        final_loss=float(m.get("final_loss", 0.0)),
                        test_accuracy_raw=float(m.get("test_accuracy_raw", 0.0)),
                        balanced_accuracy_raw=float(m.get("balanced_accuracy_raw", 0.0)),
                        mcc_raw=float(m.get("mcc_raw", 0.0)),
                        test_accuracy_readout=(
                            float(m["test_accuracy_readout"])
                            if "test_accuracy_readout" in m
                            else None
                        ),
                        balanced_accuracy_readout=(
                            float(m["balanced_accuracy_readout"])
                            if "balanced_accuracy_readout" in m
                            else None
                        ),
                        mcc_readout=float(m["mcc_readout"]) if "mcc_readout" in m else None,
                        test_accuracy_zne=(
                            float(m["test_accuracy_zne"]) if "test_accuracy_zne" in m else None
                        ),
                        balanced_accuracy_zne=(
                            float(m["balanced_accuracy_zne"])
                            if "balanced_accuracy_zne" in m
                            else None
                        ),
                        mcc_zne=float(m["mcc_zne"]) if "mcc_zne" in m else None,
                        test_accuracy_readout_zne=(
                            float(m["test_accuracy_readout_zne"])
                            if "test_accuracy_readout_zne" in m
                            else None
                        ),
                        balanced_accuracy_readout_zne=(
                            float(m["balanced_accuracy_readout_zne"])
                            if "balanced_accuracy_readout_zne" in m
                            else None
                        ),
                        mcc_readout_zne=(
                            float(m["mcc_readout_zne"]) if "mcc_readout_zne" in m else None
                        ),
                    )
        except Exception:
            cache = {}

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

    computed: list[VqcNoiseSweepRunResult] = []
    if pending:
        tasks_only = [t for _, t in pending]
        if max_workers is None or max_workers <= 1:
            iterable = tasks_only
            if use_tqdm:
                from tqdm.auto import tqdm as _tqdm

                iterable = _tqdm(tasks_only, desc="VQC noise sweep", total=len(tasks_only))
            for task in iterable:
                if verbose and not use_tqdm:
                    print(
                        f"profile={task.noise_profile} level={task.noise_level} "
                        f"shots={task.shots} seed={task.seed}"
                    )
                computed.append(
                    _single_noise_run(
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
                )
        else:
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
            computed = map_parallel_or_serial(
                tasks_only,
                _vqc_noise_worker,
                max_workers=max_workers,
                use_tqdm=use_tqdm,
                tqdm_desc="VQC noise sweep",
                initializer=_vqc_noise_pool_init,
                initargs=(cfg,),
            )
        for (idx, _), result in zip(pending, computed, strict=True):
            ordered[idx] = result

    out = VqcNoiseSweepResults(results=[r for r in ordered if r is not None])

    if mlflow_available and computed:
        import mlflow

        for r in computed:
            run_name = (
                f"{mlflow_run_prefix}|{r.noise_profile}|"
                f"lvl={r.noise_level if r.noise_level is not None else 'none'}|"
                f"shots={r.shots}|seed={r.seed}"
            )
            with mlflow.start_run(run_name=run_name):
                mlflow.log_params(
                    {
                        "pipeline": "simulated_vqc_noise",
                        "run_prefix": mlflow_run_prefix,
                        "noise_profile": r.noise_profile,
                        "noise_level": "none" if r.noise_level is None else float(r.noise_level),
                        "shots": int(r.shots),
                        "seed": int(r.seed),
                        "ansatz": r.ansatz,
                        "n_qubits": probe_vc.n_qubits,
                        "n_features": probe_vc.n_features,
                        "n_trainable": probe_vc.n_trainable,
                        "max_iter": int(max_iter),
                        "decision_rule": decision_rule,
                        "observable": observable,
                        "loss_name": loss_name,
                        "expectation_qubit": int(expectation_qubit),
                    }
                )
                metrics: dict[str, float] = {
                    "training_time": float(r.training_time),
                    "inference_time": float(r.inference_time),
                    "final_loss": float(r.final_loss),
                    "test_accuracy_raw": float(r.test_accuracy_raw),
                    "balanced_accuracy_raw": float(r.balanced_accuracy_raw),
                    "mcc_raw": float(r.mcc_raw),
                }
                if r.test_accuracy_readout is not None:
                    metrics["test_accuracy_readout"] = float(r.test_accuracy_readout)
                if r.balanced_accuracy_readout is not None:
                    metrics["balanced_accuracy_readout"] = float(r.balanced_accuracy_readout)
                if r.mcc_readout is not None:
                    metrics["mcc_readout"] = float(r.mcc_readout)
                if r.test_accuracy_zne is not None:
                    metrics["test_accuracy_zne"] = float(r.test_accuracy_zne)
                if r.balanced_accuracy_zne is not None:
                    metrics["balanced_accuracy_zne"] = float(r.balanced_accuracy_zne)
                if r.mcc_zne is not None:
                    metrics["mcc_zne"] = float(r.mcc_zne)
                if r.test_accuracy_readout_zne is not None:
                    metrics["test_accuracy_readout_zne"] = float(r.test_accuracy_readout_zne)
                if r.balanced_accuracy_readout_zne is not None:
                    metrics["balanced_accuracy_readout_zne"] = float(
                        r.balanced_accuracy_readout_zne
                    )
                if r.mcc_readout_zne is not None:
                    metrics["mcc_readout_zne"] = float(r.mcc_readout_zne)
                mlflow.log_metrics(metrics)

    return out
