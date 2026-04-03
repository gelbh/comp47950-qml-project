"""
Nim game logic, data generation, and policies for the QML project.

Submodules:
  - game: Nim rules, state representation, Nim-sum, optimal moves, play.
  - data: State enumeration, labelling, train/test split (train M≤5, test M>5), training-size subsets.
  - encoding: Angle, amplitude, binary, and IQP-style feature-map encodings.

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
    move_to_index,
    index_to_move,
    enumerate_states,
    generate_dataset,
    class_balance_table,
    split_class_balance,
    majority_baseline_accuracy,
    compute_class_weights,
    training_subsets,
    ood_split,
    normalise_states,
    prepare_experiment_data,
    all_heap_permutations,
    augment_s3,
    augment_s3_moves,
    canonical_order,
    remap_move_to_original,
    count_canonical_states,
    augmentation_stats,
)

from qml_project.nim.encoding import (  # noqa: F401
    ENCODING_CANDIDATES,
    EncodingName,
    SymmetryMode,
    PilotMetrics,
    EncodingGoNoGoCriteria,
    EncodingDecision,
    BinaryScopeCriteria,
    EncodingComparison,
    angle_parameters,
    build_angle_encoding_circuit,
    amplitude_vector,
    build_amplitude_encoding_circuit,
    binary_bits,
    build_binary_encoding_circuit,
    build_iqp_parity_feature_map,
    build_encoding_circuit,
    pilot_metrics_from_observation,
    evaluate_go_no_go,
    compare_encoding_pilots,
    evaluate_binary_scope,
    select_encodings_for_sweeps,
)
