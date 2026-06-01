# Notebooks

This directory holds the COMP47950 Quantum Machine Learning implementation notebook.

**Deliverable:** [`qml_project.ipynb`](qml_project.ipynb). The first code cell inlines imports, `SWEEP_WORKERS` / `USE_CACHE`, and imports `workflow_cache_path` from `qml_project` (implemented in `qml_project.notebook_setup`). Device refit / submission caches use [`./.workflow_cache/`](./.workflow_cache/) (created on demand).

**Environment:** For a single venv that can run the whole notebook (Qiskit, Aer, MLflow, IBM Runtime), use `make env-full` and a kernel pointed at `.venv-full` — see the repo [README](../README.md#recommended-setup-full-notebook).
