"""Interpretability helpers for classical and VQC Nim experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score

from qml_project.baselines.sweep.tasks import ClassicalSweepTask, execute_classical_sweep_task
from qml_project.circuit import VariationalClassifier, build_circuit
from qml_project.nim.data import training_subsets
from qml_project.nim.encoding import EncodingName, SymmetryMode
from qml_project.pareto_selection import filter_workflow_rows_to_winner
from qml_project.qsvm.model import evaluate_quantum_kernel_svm, fit_quantum_kernel_svm
from qml_project.training.evaluation import evaluate_classifier, train_classifier
from qml_project.training.selection import Winner
from qml_project.vqc_workflow import (
    VQC_TEST_SHOTS,
    circuit_kwargs_from_vqc_config,
    transform_states_for_vqc,
    _vqc_grid_config_by_id,
)


def classical_feature_names(*, feature_set: str = "raw", k: int = 3, m_max: int = 7) -> list[str]:
    """Return feature names matching `prepare_features` ordering."""
    names: list[str] = [f"h{i + 1}_norm" for i in range(k)]
    if feature_set == "raw":
        return names

    if feature_set in ("heap_parity", "parity"):
        names.extend([f"h{i + 1}_parity" for i in range(k)])

    if feature_set in ("pairwise_xor", "parity"):
        for i in range(k):
            for j in range(i + 1, k):
                names.append(f"h{i + 1}_xor_h{j + 1}_norm")

    if feature_set in ("bit_parity", "parity"):
        n_bits = int(np.ceil(np.log2(m_max + 1)))
        names.extend([f"nim_bit_parity_{b}" for b in range(n_bits)])

    return names


def classical_permutation_importance(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    feature_names: Sequence[str] | None = None,
    n_repeats: int = 20,
    random_state: int = 42,
    scoring: str = "balanced_accuracy",
) -> pd.DataFrame:
    """Compute permutation importances on held-out data."""
    result = permutation_importance(
        model,
        np.asarray(X_test, dtype=np.float64),
        np.asarray(y_test, dtype=np.int32),
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    names = (
        list(feature_names)
        if feature_names is not None
        else [f"f{i}" for i in range(np.asarray(X_test).shape[1])]
    )
    out = pd.DataFrame(
        {
            "feature": names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return out.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def raw_vs_parity_importance_summary(
    classical_imp_raw_df: pd.DataFrame,
    classical_imp_parity_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compact raw vs parity headline stats from two permutation-importance tables.

    Expects non-empty frames with columns ``feature``, ``importance_mean``,
    ``importance_std``, each sorted by ``importance_mean`` descending (as returned
    by :func:`classical_permutation_importance`). Returns an empty frame with the
    correct columns if either input is empty.
    """
    summary_cols: list[str] = [
        "setting",
        "top_feature",
        "top_importance_mean",
        "total_importance_mean",
    ]
    if classical_imp_raw_df.empty or classical_imp_parity_df.empty:
        return pd.DataFrame(columns=summary_cols)

    def _row(setting: str, df: pd.DataFrame) -> dict[str, str | float]:
        return {
            "setting": setting,
            "top_feature": str(df.iloc[0]["feature"]),
            "top_importance_mean": float(df.iloc[0]["importance_mean"]),
            "total_importance_mean": float(df["importance_mean"].clip(lower=0).sum()),
        }

    return pd.DataFrame(
        [
            _row("raw", classical_imp_raw_df),
            _row("parity", classical_imp_parity_df),
        ]
    )


def vqc_parameter_sensitivity(
    vc: VariationalClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    theta: np.ndarray,
    *,
    step: float = 0.05,
    shots: int = 256,
    sampler: Any | None = None,
    seed: int = 42,
    decision_rule: str = "expectation_threshold",
    expectation_qubit: int = 0,
) -> pd.DataFrame:
    """Estimate per-parameter sensitivity via +/- perturbation deltas."""
    theta_arr = np.asarray(theta, dtype=np.float64).copy()
    if theta_arr.ndim != 1:
        raise ValueError("theta must be a 1D array")

    base_eval = evaluate_classifier(
        vc,
        np.asarray(X_test, dtype=np.float64),
        np.asarray(y_test, dtype=np.int32),
        theta_arr,
        shots=shots,
        sampler=sampler,
        seed=seed,
        decision_rule=decision_rule,  # type: ignore[arg-type]
        expectation_qubit=expectation_qubit,
    )
    y_true = np.asarray(y_test, dtype=np.int32)
    base_preds = np.asarray(base_eval["predictions"], dtype=np.int32)
    base_bal_acc = float(balanced_accuracy_score(y_true, base_preds))

    rows: list[dict[str, float | int]] = []
    for i in range(theta_arr.size):
        plus = theta_arr.copy()
        minus = theta_arr.copy()
        plus[i] += step
        minus[i] -= step

        eval_plus = evaluate_classifier(
            vc,
            np.asarray(X_test, dtype=np.float64),
            y_true,
            plus,
            shots=shots,
            sampler=sampler,
            seed=seed,
            decision_rule=decision_rule,  # type: ignore[arg-type]
            expectation_qubit=expectation_qubit,
        )
        eval_minus = evaluate_classifier(
            vc,
            np.asarray(X_test, dtype=np.float64),
            y_true,
            minus,
            shots=shots,
            sampler=sampler,
            seed=seed,
            decision_rule=decision_rule,  # type: ignore[arg-type]
            expectation_qubit=expectation_qubit,
        )
        bal_plus = float(balanced_accuracy_score(y_true, np.asarray(eval_plus["predictions"], dtype=np.int32)))
        bal_minus = float(balanced_accuracy_score(y_true, np.asarray(eval_minus["predictions"], dtype=np.int32)))
        rows.append(
            {
                "param_index": int(i),
                "base_balanced_accuracy": float(base_bal_acc),
                "plus_balanced_accuracy": bal_plus,
                "minus_balanced_accuracy": bal_minus,
                "delta_plus": float(bal_plus - base_bal_acc),
                "delta_minus": float(bal_minus - base_bal_acc),
                "max_abs_delta": float(max(abs(bal_plus - base_bal_acc), abs(bal_minus - base_bal_acc))),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("max_abs_delta", ascending=False).reset_index(drop=True)


def hypothesis_verdict_balanced_accuracy_tables(
    comparison_train_size_summary: pd.DataFrame,
    *,
    metric: str = "balanced_accuracy",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build §12.1-style mean ± std tables from :func:`build_final_three_way_comparison` output."""
    empty = pd.DataFrame()
    if comparison_train_size_summary is None or comparison_train_size_summary.empty:
        return empty, empty
    s = comparison_train_size_summary.loc[
        comparison_train_size_summary["metric"].astype(str) == metric
    ].copy()
    if s.empty:
        return empty, empty

    def _fmt(mn: float, st: float) -> str:
        if not np.isfinite(mn):
            return "N/A"
        if np.isfinite(st) and st > 0.0:
            return f"{mn:.4f} ± {st:.4f}"
        return f"{mn:.4f} ± 0.0000"

    def _pivot(pipelines: list[str]) -> pd.DataFrame:
        ts_vals = sorted(pd.to_numeric(s["train_size"], errors="coerce").dropna().unique())
        rows: list[dict[str, Any]] = []
        for ts in ts_vals:
            ts_f = float(ts)
            n_disp: int | float = int(ts_f) if abs(ts_f - round(ts_f)) < 1e-9 else ts_f
            row: dict[str, Any] = {"n": n_disp}
            for pipe in pipelines:
                sub = s.loc[
                    (s["pipeline"].astype(str) == pipe)
                    & (pd.to_numeric(s["train_size"], errors="coerce") == ts_f)
                ]
                if sub.empty:
                    row[pipe] = "N/A"
                else:
                    row[pipe] = _fmt(float(sub["mean"].iloc[0]), float(sub["std"].iloc[0]))
            rows.append(row)
        return pd.DataFrame(rows)

    t1 = _pivot(["classical_parity_best", "sim_quantum_qsvm", "sim_quantum_vqc"])
    t2 = _pivot(["classical_raw_best", "sim_quantum_qsvm_heap_only", "sim_quantum_vqc_heap_only"])
    return t1, t2


def _train_subset_for_size(
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    train_size: int,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return ``(X_sub_raw, y_sub, effective_train_size)`` for refits."""
    n_full = len(X_train_raw)
    if train_size >= n_full:
        subs = training_subsets(X_train_raw, y_train, sizes=[], random_state=seed)
        ts = subs["full"]
        return ts.X, ts.y, int(ts.size)
    subs = training_subsets(X_train_raw, y_train, sizes=[int(train_size)], random_state=seed)
    ts = subs[int(train_size)]
    return ts.X, ts.y, int(ts.size)


def refit_classical_raw_best_predict_test(
    classical_raw_best_info: dict[str, Any] | None,
    *,
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    train_size: int,
    seed: int,
    M: int,
    c_svc: float = 1.0,
) -> tuple[np.ndarray, Any, str, str, str]:
    """Refit the §11 ``classical_raw_best`` configuration and return test predictions + fitted model."""
    if classical_raw_best_info is None:
        raise ValueError("classical_raw_best_info is None; run Section 11 first.")
    X_sub, y_sub, eff_size = _train_subset_for_size(
        X_train_raw, y_train, train_size, seed=seed
    )
    model_name = str(classical_raw_best_info["model"])
    feature_set = str(classical_raw_best_info["feature_set"])
    symmetry = str(classical_raw_best_info["symmetry"])
    task = ClassicalSweepTask(
        X_sub_raw=np.asarray(X_sub, dtype=np.int32),
        y_sub=np.asarray(y_sub, dtype=np.int32),
        model_name=model_name,
        feature_set=feature_set,
        symmetry=symmetry,
        train_size=eff_size,
        seed=int(seed),
        compute_win_rate=False,
        n_games_win_rate=0,
        c_svc=float(c_svc),
    )
    res = execute_classical_sweep_task(
        task,
        np.asarray(X_test_raw, dtype=np.int32),
        np.asarray(y_test, dtype=np.int32),
        M=int(M),
    )
    if res.y_pred is None:
        raise RuntimeError("Classical sweep task did not return y_pred (unexpected).")
    y_pred = np.asarray(res.y_pred, dtype=np.int32)
    from qml_project.baselines.features import FeatureSet, prepare_features
    from qml_project.baselines.models import create_models
    from qml_project.nim.data import canonical_order

    fs = cast(FeatureSet, feature_set)
    if symmetry == "canonical":
        X_test_use, _ = canonical_order(np.asarray(X_test_raw, dtype=np.int32))
        X_sub_use, _ = canonical_order(np.asarray(X_sub, dtype=np.int32))
    else:
        X_test_use = X_test_raw
        X_sub_use = X_sub
    X_sub_feat = prepare_features(X_sub_use, fs, M=M)
    models = create_models(random_state=int(seed), M=M, c_svc=float(c_svc))
    fitted = models[model_name]
    fitted.fit(np.asarray(X_sub_feat, dtype=np.float64), np.asarray(y_sub, dtype=np.int32))
    return y_pred, fitted, model_name, feature_set, symmetry


def classical_posthoc_coefficients_or_importances(
    model: Any,
    _model_name: str,
    X_test_feat: np.ndarray,
    y_test: np.ndarray,
    *,
    feature_names: Sequence[str],
    random_state: int = 42,
    perm_repeats: int = 12,
) -> pd.DataFrame:
    """Linear ``coef_`` when available, else RF importances, else permutation importance."""
    X_te = np.asarray(X_test_feat, dtype=np.float64)
    y_te = np.asarray(y_test, dtype=np.int32)
    names = list(feature_names)

    coef = getattr(model, "coef_", None)
    if coef is not None and np.asarray(coef).size > 0:
        vec = np.asarray(coef).ravel()
        if vec.size == len(names):
            return pd.DataFrame({"feature": names, "coefficient": vec}).assign(
                kind="linear_coef"
            )

    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=np.float64)
        if imp.size == len(names):
            return pd.DataFrame({"feature": names, "importance": imp}).assign(kind="rf_importance")

    imp_df = classical_permutation_importance(
        model,
        X_te,
        y_te,
        feature_names=names,
        n_repeats=int(perm_repeats),
        random_state=int(random_state),
    )
    return imp_df.assign(kind="permutation_importance")


def quantum_winners_test_predictions_interpretability(
    quantum_winners: Mapping[str, Winner],
    qsvm_workflow_df: pd.DataFrame,
    _vqc_workflow_df: pd.DataFrame,
    vqc_config_grid: Sequence[Mapping[str, Any]],
    *,
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    train_size: int,
    seed: int = 42,
    M: int = 7,
    bits_per_heap: int = 3,
    max_iter_vqc: int = 200,
    test_shots_vqc: int = VQC_TEST_SHOTS,
    class_weight: str | dict[int, float] | None = "balanced",
) -> dict[str, np.ndarray]:
    """One-shot refits of §7 QSVM/VQC winners for test-set prediction vectors (single anchor)."""
    out: dict[str, np.ndarray] = {}
    X_sub_raw, y_sub, _eff = _train_subset_for_size(
        X_train_raw, y_train, int(train_size), seed=int(seed)
    )

    w_q = quantum_winners.get("qsvm")
    if w_q is not None and qsvm_workflow_df is not None and not qsvm_workflow_df.empty:
        try:
            tmpl = filter_workflow_rows_to_winner(qsvm_workflow_df, w_q)
            tmpl = tmpl.loc[
                pd.to_numeric(tmpl["train_size"], errors="coerce") == float(train_size)
            ]
            tmpl = tmpl.loc[pd.to_numeric(tmpl["seed"], errors="coerce") == float(seed)]
            if tmpl.empty:
                tmpl = filter_workflow_rows_to_winner(qsvm_workflow_df, w_q)
                tmpl = tmpl.loc[
                    pd.to_numeric(tmpl["train_size"], errors="coerce") == float(train_size)
                ]
            if tmpl.empty:
                raise RuntimeError("No QSVM workflow row for winner/train_size.")
            row = tmpl.iloc[0]
            encoding = cast(EncodingName, str(row["encoding"]))
            symmetry = cast(SymmetryMode, str(row.get("symmetry", "none")))
            inc_ns = bool(row.get("include_nim_sum", True))
            c_sv = float(row.get("c_svc", 1.0))
            est_mode = str(row.get("estimator_mode", "exact_statevector"))
            k_back = str(row.get("kernel_backend", "manual"))
            shots = int(row.get("shots", 1024))
            bph = int(row.get("bits_per_heap", bits_per_heap))
            iqp = int(row.get("iqp_reps", 2))
            qmod, _, _ = fit_quantum_kernel_svm(
                np.asarray(X_sub_raw, dtype=np.int32),
                np.asarray(y_sub, dtype=np.int32),
                encoding=encoding,
                class_weight=class_weight,
                M=int(M),
                bits_per_heap=bph,
                iqp_reps=iqp,
                include_nim_sum=inc_ns,
                symmetry=symmetry,
                random_state=int(seed),
                c_svc=c_sv,
                estimator_mode=cast(Any, est_mode),
                kernel_backend=cast(Any, k_back),
                shots=shots,
            )
            _res, y_pred = evaluate_quantum_kernel_svm(
                qmod, np.asarray(X_test_raw, dtype=np.int32), np.asarray(y_test, dtype=np.int32)
            )
            out["qsvm"] = np.asarray(y_pred, dtype=np.int32)
        except Exception:
            pass

    w_v = quantum_winners.get("vqc")
    if w_v is not None and vqc_config_grid:
        try:
            cfg = _vqc_grid_config_by_id(vqc_config_grid, str(w_v.config_id))
            inc = bool(cfg.get("include_nim_sum", True))
            enc = str(cfg["encoding"])
            x_te = transform_states_for_vqc(
                X_test_raw,
                encoding=enc,
                include_nim_sum=inc,
                M=int(M),
                bits_per_heap=bits_per_heap,
            )
            subset_X = transform_states_for_vqc(
                X_sub_raw,
                encoding=enc,
                include_nim_sum=inc,
                M=int(M),
                bits_per_heap=bits_per_heap,
            )
            ck = circuit_kwargs_from_vqc_config(cfg)
            vc = build_circuit(**ck)
            weights, _ = train_classifier(
                vc,
                subset_X,
                np.asarray(y_sub, dtype=np.int32),
                x_te,
                np.asarray(y_test, dtype=np.int32),
                max_iter=int(max_iter_vqc),
                seed=int(seed),
                test_shots=int(test_shots_vqc),
                sampler=None,
                observable=cfg["observable"],
                decision_rule=cfg["decision_rule"],
                loss_name=cfg["loss_name"],
                expectation_qubit=0,
                verbose=False,
                log_interval=10_000,
                mlflow_experiment=None,
            )
            ev = evaluate_classifier(
                vc,
                x_te,
                np.asarray(y_test, dtype=np.int32),
                weights,
                shots=int(test_shots_vqc),
                sampler=None,
                seed=int(seed),
                decision_rule=cfg["decision_rule"],
                expectation_qubit=0,
            )
            out["vqc"] = np.asarray(ev["predictions"], dtype=np.int32)
        except Exception:
            pass

    return out


def summarize_interpretability_disagreements(
    y_true: np.ndarray,
    preds_by_name: Mapping[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-model balanced accuracy vs labels and pairwise disagreement rates."""
    y = np.asarray(y_true, dtype=np.int32)
    acc_rows: list[dict[str, Any]] = []
    for name, p in preds_by_name.items():
        pr = np.asarray(p, dtype=np.int32)
        if pr.shape[0] != y.shape[0]:
            continue
        acc_rows.append(
            {
                "model": name,
                "balanced_accuracy": float(balanced_accuracy_score(y, pr)),
                "n_test": int(y.shape[0]),
            }
        )
    acc_df = pd.DataFrame(acc_rows)
    names = [str(r["model"]) for _, r in acc_df.iterrows()]
    dis_rows: list[dict[str, Any]] = []
    pred_list = [(str(n), np.asarray(preds_by_name[n], dtype=np.int32)) for n in names]
    for i in range(len(pred_list)):
        for j in range(i + 1, len(pred_list)):
            na, pa = pred_list[i]
            nb, pb = pred_list[j]
            if pa.shape != pb.shape:
                continue
            dis_rows.append(
                {
                    "model_a": na,
                    "model_b": nb,
                    "disagreement_rate": float(np.mean(pa != pb)),
                    "n_test": int(pa.shape[0]),
                }
            )
    return acc_df, pd.DataFrame(dis_rows)


InterpretabilityCaseStudyBasis = Literal[
    "none",
    "one-sided disagreements",
    "joint failure cases",
    "non-trivial disagreement/failure cases",
]

_DEFAULT_CASE_STUDY_COLS: tuple[str, ...] = (
    "h_1",
    "h_2",
    "h_3",
    "nim_sum",
    "nim_sum_abs",
    "min_train_distance",
    "true_label",
    "pred_classical",
    "pred_qsvm",
    "disagreement_classical_vs_qsvm",
)


def select_interpretability_case_studies(
    adv_df: pd.DataFrame | None,
    *,
    disagreement_col: str = "disagreement_classical_vs_qsvm",
    case_cols: Sequence[str] | None = None,
    max_rows: int = 2,
) -> tuple[pd.DataFrame, InterpretabilityCaseStudyBasis]:
    """Pick a few adversarial rows to anchor narrative (classical vs QSVM).

    Priority: one-sided disagreements, then joint mislabels, then any non-trivial
    non-agreement with ``both_correct``.
    """
    cols = tuple(case_cols) if case_cols is not None else _DEFAULT_CASE_STUDY_COLS
    basis: InterpretabilityCaseStudyBasis = "none"
    if adv_df is None or not isinstance(adv_df, pd.DataFrame):
        return pd.DataFrame(), basis
    if disagreement_col not in adv_df.columns:
        return pd.DataFrame(), basis

    use_cols = [c for c in cols if c in adv_df.columns]
    one_sided = ("classical_only_wrong", "qsvm_only_wrong")
    joint_failures = ("both_wrong_same_label", "both_wrong_diff_label")

    candidates = adv_df.loc[adv_df[disagreement_col].isin(one_sided), use_cols]
    if not candidates.empty:
        basis = "one-sided disagreements"
    else:
        candidates = adv_df.loc[adv_df[disagreement_col].isin(joint_failures), use_cols]
        if not candidates.empty:
            basis = "joint failure cases"
        else:
            candidates = adv_df.loc[adv_df[disagreement_col] != "both_correct", use_cols]
            if not candidates.empty:
                basis = "non-trivial disagreement/failure cases"

    if candidates.empty:
        return pd.DataFrame(), "none"

    if "nim_sum_abs" in candidates.columns and "min_train_distance" in candidates.columns:
        candidates = candidates.sort_values(
            ["nim_sum_abs", "min_train_distance"],
            ascending=[True, False],
        )
    elif "nim_sum_abs" in candidates.columns:
        candidates = candidates.sort_values("nim_sum_abs", ascending=True)
    elif "min_train_distance" in candidates.columns:
        candidates = candidates.sort_values("min_train_distance", ascending=False)

    out = candidates.head(int(max_rows)).reset_index(drop=True)
    return out, basis


def interpretability_precondition_dataframe(
    *,
    classical_imp_raw_df: pd.DataFrame,
    raw_vs_parity_summary_df: pd.DataFrame,
    vqc_sens_df: pd.DataFrame,
    case_studies_df: pd.DataFrame,
) -> pd.DataFrame:
    """Tabular readiness flags for the interpretability notebook subsection."""
    checks = {
        "classical_importance_ready": int(not classical_imp_raw_df.empty),
        "raw_vs_parity_comparison_ready": int(not raw_vs_parity_summary_df.empty),
        "vqc_sensitivity_ready": int(not vqc_sens_df.empty),
        "case_studies_available": int(not case_studies_df.empty),
    }
    return pd.DataFrame([{"check": k, "ok": bool(v), "detail": None} for k, v in checks.items()])
