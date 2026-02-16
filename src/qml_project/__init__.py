"""
Shared code for COMP47950 Quantum Machine Learning project.
Use for data loading, metrics, circuit helpers, or other logic reused across the notebook.
"""

from qml_project.circuit import (  # noqa: F401
    VariationalClassifier,
    build_circuit,
    bitstring_to_class_map,
    counts_to_class_probs,
    softmax_nll_loss,
    batch_loss,
    predict_from_probs,
    predict_batch,
)

from qml_project.training import (  # noqa: F401
    TrainingHistory,
    ExperimentResult,
    MultiSeedSummary,
    shots_for_eval,
    evaluate_circuit,
    train_classifier,
    evaluate_classifier,
    run_multi_seed_experiment,
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
