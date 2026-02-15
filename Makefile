PYTHON ?= python3.10
UV     := uv

.PHONY: env-qiskit env-pennylane env-device clean-envs

env-qiskit:
	$(UV) venv .venv-qiskit --python $(PYTHON)
	VIRTUAL_ENV=.venv-qiskit $(UV) pip install -e ".[qiskit]"

env-pennylane:
	$(UV) venv .venv-pennylane --python $(PYTHON)
	VIRTUAL_ENV=.venv-pennylane $(UV) pip install -e ".[pennylane]"

env-device:
	$(UV) venv .venv-device --python $(PYTHON)
	VIRTUAL_ENV=.venv-device $(UV) pip install -e ".[device]"

clean-envs:
	rm -rf .venv-qiskit .venv-pennylane .venv-device
