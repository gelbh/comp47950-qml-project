# Quantum Machine Learning (COMP47950)

![Course](https://img.shields.io/badge/Course-COMP47950-blue)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![MLflow](https://img.shields.io/badge/MLflow-tracking-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-demo-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

This submission compares **classical**, **simulated QML** (VQC and QSVM), and **inference-only IBM Quantum** evaluation on a Nim classification task. The main deliverable is the implementation notebook and report; optional extras include a Streamlit demo.

## What is in this bundle

- **`src/qml_project/`** — importable package (circuits, training, baselines, device helpers, Nim utilities).
- **`notebooks/`** — primary submission: [`notebooks/qml_project.ipynb`](notebooks/qml_project.ipynb) (implementation and report).
- **`apps/nim_demo/`** — Streamlit demonstration (see [`apps/nim_demo/README.md`](apps/nim_demo/README.md) for layout and behaviour).

Also included: **`Makefile`**, **`pyproject.toml`**, **`uv.lock`**.

## Prerequisites

- **Python 3.10** (matches the default in the `Makefile`).
- **[uv](https://docs.astral.sh/uv/)** — install from the vendor instructions, for example:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Recommended setup (full notebook)

From the repository root, create the environment that includes Jupyter, MLflow, Qiskit, Aer, and IBM Runtime, then start the notebook:

```bash
make env-full
make run-notebook-full
```

The `qml_project` package is installed from `src/` into that environment.

Lighter dependency sets (Qiskit-only, device-only) exist as other `make env-*` targets in the `Makefile` if you need them.

### Optional: Jupyter kernel for `.venv-full`

If you use Jupyter outside the `make run-notebook-full` flow, register a kernel (then pick **Python (qml-full)** in the kernel menu):

```bash
UV_PROJECT_ENVIRONMENT=.venv-full uv run python -m ipykernel install --user \
  --name=qml-full --display-name="Python (qml-full)"
```

## Streamlit demo (same venv as the full notebook)

`make run-demo` in the `Makefile` uses a different virtualenv (`.venv-qiskit`). If you only created **`.venv-full`**, run the demo with:

```bash
UV_PROJECT_ENVIRONMENT=.venv-full uv run streamlit run apps/nim_demo/app.py
```

Alternatively, run `make env-qiskit` and then `make run-demo`.

## MLflow UI (optional)

After `make env-full`, you can browse local experiment data (if present) with:

```bash
UV_PROJECT_ENVIRONMENT=.venv-full uv run mlflow ui
```

then open [http://localhost:5000](http://localhost:5000). The graded narrative lives in the notebook outputs; MLflow is not required to read the report.
