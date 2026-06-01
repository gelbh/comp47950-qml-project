PYTHON ?= python3.10
UV     := uv

.PHONY: env-full env-qiskit env-device run-notebook-full run-notebook-qiskit run-notebook-device mlflow-ui-qiskit run-demo clean-envs

# Main notebook stack: Jupyter, MLflow, Qiskit, Aer, IBM Runtime.
env-full:
	UV_PROJECT_ENVIRONMENT=.venv-full $(UV) sync --python $(PYTHON) --group full-notebook

env-qiskit:
	UV_PROJECT_ENVIRONMENT=.venv-qiskit $(UV) sync --python $(PYTHON) --group qiskit

env-device:
	UV_PROJECT_ENVIRONMENT=.venv-device $(UV) sync --python $(PYTHON) --group device

run-notebook-full:
	UV_PROJECT_ENVIRONMENT=.venv-full $(UV) run jupyter notebook notebooks/qml_project.ipynb

run-notebook-qiskit:
	UV_PROJECT_ENVIRONMENT=.venv-qiskit $(UV) run jupyter notebook notebooks/qml_project.ipynb

run-notebook-device:
	UV_PROJECT_ENVIRONMENT=.venv-device $(UV) run jupyter notebook notebooks/qml_project.ipynb

mlflow-ui-qiskit:
	UV_PROJECT_ENVIRONMENT=.venv-qiskit $(UV) run mlflow ui

run-demo:
	UV_PROJECT_ENVIRONMENT=.venv-qiskit $(UV) run streamlit run apps/nim_demo/app.py

clean-envs:
	rm -rf .venv-full .venv-qiskit .venv-device
