"""Entry point for the multipage Nim QML demo.

Run from the project root with::

    make run-demo
    # or: UV_PROJECT_ENVIRONMENT=.venv-qiskit uv run streamlit run apps/nim_demo/app.py

This file owns two global concerns and nothing else:

* :func:`st.set_page_config` — called once for the whole app.
* :func:`st.navigation` — declares the Play page and the eleven Learn pages.

Play orchestration lives in ``play.py`` (thin entry), ``play_layout.py``,
``play_panel.py``, ``play_session.py``, and ``play_turns.py``; ``pages/01_play.py``
calls ``play.render()``.
``apps/nim_demo/pages/`` that imports from the ``viz`` package, ``loaders``,
``content``, and the reusable ``qml_project`` package.
"""

from __future__ import annotations

from pathlib import Path

from _path_setup import ensure_demo_path

ensure_demo_path()

import streamlit as st

_APP_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Nim QML demo",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES_DIR = _APP_DIR / "pages"

play_page = st.Page(
    str(PAGES_DIR / "01_play.py"),
    title="Play",
    icon=":material/sports_esports:",
    default=True,
)
problem_page = st.Page(
    str(PAGES_DIR / "02_problem.py"),
    title="The problem",
    icon=":material/extension:",
)
data_page = st.Page(
    str(PAGES_DIR / "03_data.py"),
    title="Data",
    icon=":material/dataset:",
)
parity_features_page = st.Page(
    str(PAGES_DIR / "04_parity_features.py"),
    title="Classical features",
    icon=":material/tune:",
)
encoding_page = st.Page(
    str(PAGES_DIR / "05_encoding.py"),
    title="Encoding",
    icon=":material/input:",
)
vqc_page = st.Page(
    str(PAGES_DIR / "06_vqc.py"),
    title="VQC architecture",
    icon=":material/memory:",
)
qsvm_page = st.Page(
    str(PAGES_DIR / "07_qsvm.py"),
    title="QSVM architecture",
    icon=":material/hub:",
)
classical_page = st.Page(
    str(PAGES_DIR / "08_classical.py"),
    title="Classical baselines",
    icon=":material/psychology:",
)
training_page = st.Page(
    str(PAGES_DIR / "09_training.py"),
    title="Training",
    icon=":material/model_training:",
)
noise_page = st.Page(
    str(PAGES_DIR / "10_noise_device.py"),
    title="Noise & device",
    icon=":material/sensors:",
)
results_page = st.Page(
    str(PAGES_DIR / "11_results.py"),
    title="Results",
    icon=":material/leaderboard:",
)
faq_page = st.Page(
    str(PAGES_DIR / "12_faq.py"),
    title="FAQ",
    icon=":material/help:",
)

# Learn order: setup (problem, data, classical features, encoding) → classical baseline →
# quantum pipelines → how we train → evaluation (noise, results) → FAQ.
nav = st.navigation(
    {
        "Demo": [play_page],
        "Learn": [
            problem_page,
            data_page,
            parity_features_page,
            encoding_page,
            classical_page,
            vqc_page,
            qsvm_page,
            training_page,
            noise_page,
            results_page,
            faq_page,
        ],
    }
)
nav.run()
