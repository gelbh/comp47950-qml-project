"""Play page: sidebar, board column, and title."""

from __future__ import annotations

import time

import streamlit as st

import viz  # type: ignore[import-not-found]
from qml_project.nim.game import NimState, is_terminal, nim_sum

import play_session as ps
import play_turns as pt
from play_config import (
    CLASSICAL_CHOICES,
    MODES,
    PIPELINES,
    STARTING_PRESETS,
    classical_short,
)


def render_variant_inline(slot: str, pipeline: str) -> None:
    """Compact train-size / classifier controls inside the sidebar column."""
    from loaders import (  # type: ignore[import-not-found]
        list_qsvm_payload_sizes,
        list_vqc_payload_sizes,
        load_qsvm_payload,
        load_vqc_payload,
    )

    settings = st.session_state.settings
    variant = settings[f"variant_{slot}"]
    if pipeline == "VQC":
        sizes = list_vqc_payload_sizes()
        if not sizes:
            st.info("No VQC payloads found.")
            return
        if variant.get("vqc_size") not in sizes:
            variant["vqc_size"] = sizes[0]
        variant["vqc_size"] = st.selectbox(
            "Train size (n)",
            options=sizes,
            index=sizes.index(variant["vqc_size"]),
            key=f"vqc_size_{slot}",
        )
        variant["shots"] = st.slider(
            "Shots per move",
            min_value=128,
            max_value=2048,
            step=128,
            value=int(variant.get("shots", settings.get("shots", 512))),
            key=f"vqc_shots_{slot}",
        )
        try:
            _pl = load_vqc_payload(int(variant["vqc_size"]))
            _fk = dict(_pl.feature_kwargs) if _pl.feature_kwargs else {}
            _ns = _fk.get("include_nim_sum", "—")
            st.caption(
                f"`config_id`: `{_pl.config_id}` · **encoding** `{_pl.encoding}` · "
                f"**include_nim_sum** `{_ns}`"
            )
        except Exception:
            pass
    elif pipeline == "QSVM":
        sizes = list_qsvm_payload_sizes()
        if not sizes:
            st.info("No QSVM payloads found.")
            return
        if variant.get("qsvm_size") not in sizes:
            variant["qsvm_size"] = sizes[0]
        variant["qsvm_size"] = st.selectbox(
            "Train size (n)",
            options=sizes,
            index=sizes.index(variant["qsvm_size"]),
            key=f"qsvm_size_{slot}",
        )
        try:
            _pq = load_qsvm_payload(int(variant["qsvm_size"]))
            st.caption(
                f"`variant_id`: `{_pq.variant_id}` · **include_nim_sum** "
                f"`{_pq.include_nim_sum}`"
            )
        except Exception:
            pass
    elif pipeline == "Classical":
        current = variant.get("classical_name", CLASSICAL_CHOICES[0])
        if current not in CLASSICAL_CHOICES:
            current = CLASSICAL_CHOICES[0]
            variant["classical_name"] = current
        short = classical_short(current)
        with st.popover(
            f"Classifier · **{short}**",
            help="Open for full model names, same as the training notebooks.",
            use_container_width=True,
        ):
            variant["classical_name"] = st.selectbox(
                "Model (full name)",
                options=CLASSICAL_CHOICES,
                index=CLASSICAL_CHOICES.index(current),
                key=f"classical_name_{slot}",
            )


def sidebar() -> None:
    st.sidebar.header("Match setup")
    settings = st.session_state.settings
    legacy_shots = int(settings.get("shots", 512))
    for vk in ("variant_opp", "variant_a", "variant_b"):
        vdict = settings.get(vk)
        if isinstance(vdict, dict):
            vdict.setdefault("shots", legacy_shots)

    settings["mode"] = st.sidebar.radio(
        "Mode",
        options=MODES,
        index=MODES.index(settings["mode"]),
        horizontal=True,
    )

    if settings["mode"] == "You vs Model":
        settings["opponent"] = st.sidebar.selectbox(
            "Play against",
            options=list(PIPELINES),
            index=list(PIPELINES).index(settings["opponent"]),
        )
        with st.sidebar:
            render_variant_inline("opp", settings["opponent"])
        settings["human_first"] = st.sidebar.toggle(
            "You go first", value=settings["human_first"]
        )
    else:
        col_a, col_b = st.sidebar.columns(2, gap="small")
        with col_a:
            settings["model_a"] = st.selectbox(
                "Model A",
                options=list(PIPELINES),
                index=list(PIPELINES).index(settings["model_a"]),
                key="model_a_select",
            )
            render_variant_inline("a", settings["model_a"])
        with col_b:
            settings["model_b"] = st.selectbox(
                "Model B",
                options=list(PIPELINES),
                index=list(PIPELINES).index(settings["model_b"]),
                key="model_b_select",
            )
            render_variant_inline("b", settings["model_b"])
        settings["a_first"] = st.sidebar.toggle("A goes first", value=settings["a_first"])
        settings["auto_play"] = st.sidebar.toggle(
            "Auto-play",
            value=settings["auto_play"],
            help="When on, each model move happens automatically with the delay below.",
        )
        settings["move_delay"] = st.sidebar.slider(
            "Seconds between moves",
            min_value=0.5,
            max_value=10.0,
            step=0.5,
            value=float(settings["move_delay"]),
        )

    preset_keys = list(STARTING_PRESETS)
    settings["starting_preset"] = st.sidebar.selectbox(
        "Starting position",
        options=preset_keys,
        index=preset_keys.index(settings["starting_preset"]),
    )

    if st.sidebar.button("Start / reset game", type="primary", width="stretch"):
        ps.reset_game()


def status_caption() -> str:
    g = st.session_state.game
    if g["winner"] is not None or is_terminal(g["state"]):
        return "Game over"
    cur = ps.current_slot()
    if cur == "human":
        return "Your turn — click a stone"
    return f"{ps.slot_label(cur)}'s turn"


def advance_model_turn() -> None:
    cur = ps.current_slot()
    state = st.session_state.game["state"]
    if cur == "human":
        return
    move = pt.pick_slot_move(cur, state)
    ps.apply_move(move, slot=cur)
    st.rerun()


def board_and_input() -> None:
    game = st.session_state.game
    state: NimState = game["state"]
    max_heap = max(max(game["initial_state"]), 7)
    cur = ps.current_slot()

    st.markdown(f"### {status_caption()}")

    interactive = (
        cur == "human"
        and game["winner"] is None
        and not is_terminal(state)
    )
    fig = viz.board_figure(state, max_heap_size=max_heap, interactive=interactive)
    chart_kwargs: dict = dict(
        width="stretch",
        theme=None,
        key=f"board_t{game['turn_number']}",
    )
    if interactive:
        chart_kwargs["on_select"] = "rerun"
        chart_kwargs["selection_mode"] = ["points"]
    board_event = st.plotly_chart(fig, **chart_kwargs)
    if interactive and board_event and getattr(board_event, "selection", None):
        pts = board_event.selection.get("points") or []
        if pts:
            cd = pts[0].get("customdata") or []
            if len(cd) >= 2:
                heap_idx = int(cd[0])
                amount = int(cd[1])
                if 1 <= amount <= int(state[heap_idx]):
                    ps.apply_move((heap_idx, amount), slot="human")
                    st.rerun()

    ns = nim_sum(state)
    ns_msg = (
        f"Nim-sum = **{ns}** · "
        + (
            "current player can force a win"
            if ns != 0
            else "current player is losing against optimal play"
        )
    )
    st.caption(ns_msg)
    if any(h > 5 for h in state):
        st.caption("Heap > 5: outside the training distribution (M ≤ 5).")

    if game["winner"] is not None:
        winner = game["winner"]
        if winner == "human":
            st.success("You took the last stone. You win!")
        else:
            st.success(f"{ps.slot_label(winner)} took the last stone and wins.")
        return
    if is_terminal(state):
        return

    if cur != "human":
        settings = st.session_state.settings
        auto = settings["mode"] == "Model vs Model" and settings.get("auto_play", False)
        if auto:
            delay = float(settings.get("move_delay", 2.0))
            st.caption(
                f"Auto-play on: {ps.slot_label(cur)} will move in {delay:.1f}s. "
                "Read the right-hand panel to see why."
            )
        else:
            if st.button(
                f"Advance — {ps.slot_label(cur)} to move",
                type="primary",
                width="stretch",
                key=f"advance_{game['turn_number']}",
            ):
                advance_model_turn()


def title() -> str:
    g = st.session_state.game
    settings = st.session_state.settings
    if settings["mode"] == "You vs Model":
        return f"Nim — you vs {ps.slot_label('opp')}"
    p0, p1 = g["players"]
    return f"Nim — {ps.slot_label(p0)} vs {ps.slot_label(p1)}"


def maybe_auto_advance() -> None:
    """If the current turn is a model in auto-play, sleep then apply a move."""
    game = st.session_state.game
    settings = st.session_state.settings
    if settings["mode"] != "Model vs Model":
        return
    if not settings.get("auto_play", False):
        return
    if game["winner"] is not None or is_terminal(game["state"]):
        return
    cur = ps.current_slot()
    if cur == "human":
        return
    time.sleep(float(settings.get("move_delay", 2.0)))
    move = pt.pick_slot_move(cur, game["state"])
    ps.apply_move(move, slot=cur)
    st.rerun()
