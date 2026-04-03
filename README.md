# Quantum Machine Learning (COMP47950)

![COMP47950](https://img.shields.io/badge/Course-COMP47950-blue)
[![Python](https://img.shields.io/badge/Python-3.x-informational)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-purple)](https://mlflow.org)

Compare **classical**, **simulated QML**, and **quantum-device (inference-only)** pipelines on a simple ML task. Deliverables: midterm presentation, implementation notebook & report, demonstration.

## Development setup

The project uses **three separate uv-managed environments** to avoid dependency conflicts between quantum frameworks:

| Environment       | Purpose                                      | Install target       |
| ----------------- | -------------------------------------------- | -------------------- |
| `.venv-qiskit`    | Qiskit simulation (training + evaluation)    | `make env-qiskit`    |
| `.venv-pennylane` | PennyLane simulation (alternative framework) | `make env-pennylane` |
| `.venv-device`    | Real-device inference via IBM Quantum        | `make env-device`    |

### Prerequisites

Install [uv](https://docs.astral.sh/uv/) (fast Python package manager):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Create environments

From the repo root, use `make` to create any (or all) environments. Each target runs `uv sync` with the corresponding dependency group and writes into the named environment via `UV_PROJECT_ENVIRONMENT`:

```bash
make env-qiskit      # Qiskit simulation
make env-pennylane   # PennyLane simulation
make env-device      # Real-device inference (IBM Quantum)
```

Core + notebook + tracking dependencies are installed by default, and the framework-specific group is added per target.

### Usage

Run commands through uv without manual activation:

```bash
make run-notebook-qiskit
```

The `qml_project` package is available in all three environments.

### Experiment tracking with MLflow

This project uses MLflow to track design-space exploration runs, enabling systematic comparison of circuit configurations.

**View tracked experiments:**

```bash
make mlflow-ui-qiskit
```

Open [http://localhost:5000](http://localhost:5000) to browse runs, compare configurations, and view metrics.

**Note:** MLflow is used during exploration but not required to view the final deliverable. To fully reproduce, include the `mlruns/` directory when archiving the project.

### Cleanup

```bash
make clean-envs      # Remove all three venvs
```
