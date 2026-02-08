# Quantum Machine Learning (COMP47950)

[![COMP47950](https://img.shields.io/badge/Course-COMP47950-blue)](#)
[![Python](https://img.shields.io/badge/Python-3.x-informational)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-purple)](https://mlflow.org)

Compare **classical**, **simulated QML**, and **quantum-device (inference-only)** pipelines on a simple ML task. Deliverables: midterm presentation, implementation notebook & report, demonstration.

## Development setup

From the repo root, install the project in editable mode with MLflow support:

```bash
pip install -e ".[mlflow]"
```

Then run the main notebook from `notebooks/` (e.g. `jupyter notebook notebooks/qml_project.ipynb`). The `qml_project` package is used for datasets, preprocessing, and baselines.
