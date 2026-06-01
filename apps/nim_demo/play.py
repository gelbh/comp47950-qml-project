"""Interactive Nim match page (Streamlit).

Two modes:

* **You vs Model** — click a stone on the board to play; the model
  advances when you press the step button.
* **Model vs Model** — pin two models against each other (even two
  instances of the same pipeline with different train sizes). Auto-play
  at a configurable tempo so the audience can read each explanation.

The right-hand panel always tracks the model that is about to move
(or just moved), so you never see cross-wired visualisations.

This module exposes :func:`render`, which ``pages/01_play.py`` calls.
``app.py`` owns ``st.set_page_config`` and navigation; orchestration is
split across ``play_layout``, ``play_panel``, ``play_session``, and
``play_turns``.
"""

from __future__ import annotations

from _path_setup import ensure_demo_path

ensure_demo_path()

import streamlit as st

import play_layout
import play_panel
import play_session as ps


def render() -> None:
    """Render the Play page. Called by ``pages/01_play.py``."""
    ps.init_state()
    play_layout.sidebar()

    st.title(play_layout.title())
    st.caption(
        "Normal-play Nim: whoever takes the last stone wins. The panel on "
        "the right always reflects the model that is about to move."
    )

    left, right = st.columns([0.42, 0.58], gap="large")
    with left:
        play_layout.board_and_input()
    with right:
        play_panel.render_right_panel()

    play_layout.maybe_auto_advance()
