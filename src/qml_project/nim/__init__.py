"""
Nim game logic, data generation, and policies for the QML project.

Submodules:
  - game: Nim rules, state representation, Nim-sum, optimal moves, play.
  - data: State enumeration, labelling, train/test split (train M≤5, test M>5), training-size subsets.
  - encoding: Angle, amplitude, and binary encodings.

Import public symbols from the top-level subpackage::

    from qml_project.nim import NimState, nim_sum, optimal_move, play_game
    from qml_project.nim import generate_dataset, prepare_experiment_data
"""

from qml_project.nim.game import (  # noqa: F401
    NimState,
    NimMove,
    Policy,
    GameRecord,
    nim_sum,
    is_terminal,
    is_winning,
    legal_moves,
    apply_move,
    optimal_move,
    random_policy,
    optimal_policy,
    play_game,
    play_many,
)

from qml_project.nim.data import (  # noqa: F401
    NimDataset,
    TrainSubset,
    OODSplit,
    NimExperimentData,
    enumerate_states,
    generate_dataset,
    class_balance_table,
    split_class_balance,
    majority_baseline_accuracy,
    compute_class_weights,
    training_subsets,
    ood_split,
    normalise_states,
    angle_rad_from_heaps,
    prepare_experiment_data,
    all_heap_permutations,
    augment_s3,
    canonical_order,
    count_canonical_states,
    augmentation_stats,
)

from qml_project.nim.state_utils import state_tuple_from_array  # noqa: F401

from qml_project.nim.encoding import (  # noqa: F401
    ENCODING_CANDIDATES,
    EncodingName,
    SymmetryMode,
    DualPipelineEncodingMetrics,
    DualPipelineGateCriteria,
    DualPipelineEncodingDecision,
    BinaryScopeCriteria,
    angle_parameters,
    angle_features_matrix,
    build_angle_encoding_circuit,
    amplitude_vector,
    build_amplitude_encoding_circuit,
    binary_angle_features_matrix,
    binary_bits,
    build_binary_encoding_circuit,
    build_encoding_circuit,
    evaluate_dual_pipeline_go_no_go,
    select_dual_pipeline_encodings,
    rank_dual_pipeline_encodings,
)

__all__ = [
    # game
    "NimState",
    "NimMove",
    "Policy",
    "GameRecord",
    "nim_sum",
    "is_terminal",
    "is_winning",
    "legal_moves",
    "apply_move",
    "optimal_move",
    "random_policy",
    "optimal_policy",
    "play_game",
    "play_many",
    # data
    "NimDataset",
    "TrainSubset",
    "OODSplit",
    "NimExperimentData",
    "enumerate_states",
    "generate_dataset",
    "class_balance_table",
    "split_class_balance",
    "majority_baseline_accuracy",
    "compute_class_weights",
    "training_subsets",
    "ood_split",
    "normalise_states",
    "angle_rad_from_heaps",
    "prepare_experiment_data",
    "all_heap_permutations",
    "augment_s3",
    "canonical_order",
    "count_canonical_states",
    "augmentation_stats",
    # state utils
    "state_tuple_from_array",
    # encoding
    "ENCODING_CANDIDATES",
    "EncodingName",
    "SymmetryMode",
    "DualPipelineEncodingMetrics",
    "DualPipelineGateCriteria",
    "DualPipelineEncodingDecision",
    "BinaryScopeCriteria",
    "angle_parameters",
    "angle_features_matrix",
    "build_angle_encoding_circuit",
    "amplitude_vector",
    "build_amplitude_encoding_circuit",
    "binary_angle_features_matrix",
    "binary_bits",
    "build_binary_encoding_circuit",
    "build_encoding_circuit",
    "evaluate_dual_pipeline_go_no_go",
    "select_dual_pipeline_encodings",
    "rank_dual_pipeline_encodings",
]
