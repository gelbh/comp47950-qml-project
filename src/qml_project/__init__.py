"""
Shared code for COMP47950 Quantum Machine Learning project.

General-purpose VQC, training, design-space, and baseline modules are exported
here.  Nim-specific game logic, data generation, and policies are in the
``qml_project.nim`` subpackage.
"""

from qml_project.circuit import (  # noqa: F401
    AnsatzName,
    VariationalClassifier,
    build_circuit,
    bitstring_to_class_map,
    counts_to_class_probs,
    counts_to_z_expectation,
    softmax_nll_loss,
    batch_loss,
    predict_from_probs,
    predict_batch,
)

from qml_project.training import (  # noqa: F401
    MeasurementObservable,
    DecisionRule,
    LossName,
    TrainingHistory,
    ExperimentResult,
    MultiSeedSummary,
    SimulatedVQCRunResult,
    SimulatedVQCSweepResults,
    shots_for_eval,
    evaluate_circuit,
    evaluate_circuit_outputs,
    vqc_policy,
    evaluate_vqc_win_rate,
    train_classifier,
    evaluate_classifier,
    run_multi_seed_experiment,
    bootstrap_mean_ci,
    paired_cohens_d,
    rank_biserial_from_deltas,
    sample_efficiency_stat_tests,
    fit_power_law_learning_curve,
    run_simulated_vqc_ood_sweep,
    create_depolarizing_noise_model,
    create_noisy_sampler,
)

from qml_project.design_space import (  # noqa: F401
    CircuitConfig,
    DesignSpaceResult,
    run_design_space,
    summarize_results,
    select_device_circuits,
    qubit_sweep_configs,
    depth_sweep_configs,
    cz_sweep_configs,
)

from qml_project.expressibility import (  # noqa: F401
    ExpressibilityMetrics,
    EntanglingMetrics,
    estimate_expressibility,
    estimate_entangling_capability,
    gradient_variance_vs_depth,
    compare_ansatz_expressibility,
)

from qml_project.baselines import (  # noqa: F401
    ClassicalResult,
    SweepResults,
    FeatureSet,
    ABLATION_FEATURE_SETS,
    ABLATION_FEATURE_SETS_NO_RAW,
    FEATURE_SET_DESCRIPTIONS,
    prepare_features,
    engineer_parity_features,
    create_models,
    evaluate_model,
    angle_encoding_kernel,
    centered_kernel_alignment,
    label_kernel_binary,
    kernel_class_separation,
    compare_kernels_for_nim,
    model_policy,
    evaluate_win_rate,
    run_classical_sweep,
    run_baseline,
)
