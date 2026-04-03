PYTHON ?= python3.10
UV     := uv

.PHONY: env-qiskit env-pennylane env-device run-notebook-qiskit run-notebook-pennylane run-notebook-device mlflow-ui-qiskit clean-envs

env-qiskit:
	UV_PROJECT_ENVIRONMENT=.venv-qiskit $(UV) sync --python $(PYTHON) --group qiskit

env-pennylane:
	UV_PROJECT_ENVIRONMENT=.venv-pennylane $(UV) sync --python $(PYTHON) --group pennylane

env-device:
	UV_PROJECT_ENVIRONMENT=.venv-device $(UV) sync --python $(PYTHON) --group device

run-notebook-qiskit:
	UV_PROJECT_ENVIRONMENT=.venv-qiskit $(UV) run jupyter notebook notebooks/qml_project.ipynb

run-notebook-pennylane:
	UV_PROJECT_ENVIRONMENT=.venv-pennylane $(UV) run jupyter notebook notebooks/qml_project.ipynb

run-notebook-device:
	UV_PROJECT_ENVIRONMENT=.venv-device $(UV) run jupyter notebook notebooks/qml_project.ipynb

mlflow-ui-qiskit:
	UV_PROJECT_ENVIRONMENT=.venv-qiskit $(UV) run mlflow ui

clean-envs:
	rm -rf .venv-qiskit .venv-pennylane .venv-device
