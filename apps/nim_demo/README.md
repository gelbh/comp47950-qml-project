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

## Publishing (Streamlit Community Cloud)

The demo is deployed from this repo on [Streamlit Community Cloud](https://share.streamlit.io/). Demo data lives in **`notebooks/.workflow_cache/`** (tracked in git; parquets, VQC/QSVM payloads, and optional §10 device result pickles).

**Live app:** [https://nim-quantum-ml.streamlit.app/](https://nim-quantum-ml.streamlit.app/)

### One-time deploy (maintainer)

1. Push the repo to **public GitHub** (includes `requirements.txt`, `.streamlit/config.toml`, and `notebooks/.workflow_cache/`).
2. At [share.streamlit.io](https://share.streamlit.io/) → **New app** → select the repo.
3. **Main file path:** `apps/nim_demo/app.py`
4. **Python version:** **3.10** (Advanced settings at deploy time — do not leave the default 3.14; Qiskit and this project target 3.10).
5. Deploy (installs **`apps/nim_demo/requirements.txt`**, which includes root `requirements.txt` with `-e .`). Qiskit is in `pyproject.toml` `[project].dependencies`. No secrets required.

### Refresh `requirements.txt` after dependency changes

```bash
UV_PROJECT_ENVIRONMENT=.venv-qiskit uv export --group qiskit --no-default-groups --no-dev -o requirements.txt
```

The export includes project dependencies plus Qiskit (not the notebook/MLflow groups). It should start with `-e .` (editable install of this package). Reboot the Cloud app after pushing.

### Refresh demo data after re-running the notebook

Re-run cells that print `nim_demo: wrote ...`, plus §8.5 (payload pickles) and §10 (device results) as needed. Commit updated files under `notebooks/.workflow_cache/`, push, then **Reboot app** on Streamlit Cloud.

### Post-deploy smoke test

| Area                                   | Check                                      |
| -------------------------------------- | ------------------------------------------ |
| Play → VQC / QSVM                      | Train-size dropdown populated; a move runs |
| Learn → Classical, VQC, QSVM, Training | Plots/tables from parquets                 |
| Learn → Noise/Device, Results          | Device traces from `*_device_result_*.pkl` |
