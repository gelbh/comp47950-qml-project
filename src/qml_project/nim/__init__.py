"""
Nim game logic, data generation, and policies for the QML project.

Submodules:
  - game: Nim rules, state representation, Nim-sum, optimal moves, play.
  - data: State enumeration, labelling, IID/OOD splits, training-size subsets.

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
    SplitArrays,
    TrainSubset,
    OODSplit,
    NimExperimentData,
    move_to_index,
    index_to_move,
    enumerate_states,
    generate_dataset,
    class_balance_table,
    iid_split,
    training_subsets,
    ood_split,
    normalise_states,
    prepare_experiment_data,
)
