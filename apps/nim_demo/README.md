# Interactive Nim demo

Streamlit app for the COMP47950 demo station: play normal-play Nim (or
watch two models), with a turn-by-turn explanation panel. **Learn**
covers common visitor questions so the presenter does not have to cover
everything live.

## Run

Use the Qiskit env (not the default `uv run` env):

```bash
make run-demo
```

Equivalent:

```bash
UV_PROJECT_ENVIRONMENT=.venv-qiskit uv run streamlit run apps/nim_demo/app.py
```

Create `.venv-qiskit` once with `make env-qiskit` if needed. Default URL:
`http://localhost:8501`.

## Navigation

Sidebar: **Demo → Play** (live game) and **Learn** (topic pages, one
screen each, light interactivity).

Learn order follows `app.py`: problem → data → classical features →
encoding → classical baselines → VQC → QSVM → training → noise/device →
results → FAQ. Sections on encoding / VQC / QSVM / results / Play call
out **encoding × `include_nim_sum`** where it matters. Data comes from
`notebooks/.workflow_cache/`; no MLflow server, training, or device jobs
at runtime.

## Play page

**Mode:** **You vs Model** (pick VQC, QSVM, or Classical; **Advance**
before the model moves) or **Model vs Model** (two slots with separate
train size / classifier; auto-play with **Seconds between moves**, or
manual advance). **Start / reset game** after changing settings.

Board: click a stone to take it and everything to its right in that heap.
Nim-sum and OOD hints (`M ≤ 5` training cutoff) sit under the board.

**Right panel:** tabs depend on whose turn it is. Always **Decision**
(move, confidence, optimum agreement, inference ms), **Compare** (all
pipelines + inference), **History**. Per pipeline: VQC (circuit,
encoding heatmap, class probs), QSVM (kernel row, SV contributions),
Classical (features, probs when available).

**Models:** VQC/QSVM from `notebooks/.workflow_cache/` device payloads
(`vqc_device_payload_n*.pkl`, `qsvm_device_payload_n*.pkl`); train size,
shots, `config_id` / `variant_id` / **`include_nim_sum`** in captions.
Classical: Logistic Regression, SVM (RBF), or Random Forest, fit at
startup on heaps ≤ 5.

## File layout

- `app.py` — entry, `st.set_page_config`, `st.navigation`
- `_path_setup.py` — `sys.path` for flat imports
- `play.py` — `render()` → `pages/01_play.py`
- `play_layout.py` — sidebar, board, auto-advance
- `play_panel.py` — explanation tabs
- `play_session.py`, `play_turns.py` — state, cache, agreement log
- `pages/NN_*.py` — Learn pages
- `engine.py` — per-pipeline turn explanations
- `viz/` — Plotly / Matplotlib (`import viz`)
- `loaders.py` — cache + sklearn; refresh via notebook Section 07 (`qml_project.nim_demo_export`)
- `content.py` — titles, blurbs, FAQ

## Notes

Inference is local statevector simulation (no live hardware; device bars
use cached pickles). Auto-play uses `time.sleep` so the page pauses
briefly on purpose between moves.
