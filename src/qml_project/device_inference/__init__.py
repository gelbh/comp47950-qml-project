"""Device inference helpers for VQC and QSVM winners.

Section 10 submits each winner to IBM Runtime. The scope is split by
pipeline:

- :mod:`qml_project.device_inference.vqc` — :class:`VQCDevicePayload` plus
  refit, pub construction, and counts decoding.
- :mod:`qml_project.device_inference.qsvm` — :class:`QSVMDevicePayload`
  plus refit, overlap-circuit construction, and counts decoding.
- :mod:`qml_project.device_inference.refit_sweep` —
  :func:`~qml_project.device_inference.refit_sweep.run_device_refit_sweep_and_cache`
  for Section 8.5 multi-anchor refits and workflow-cache pickles.
- :mod:`qml_project.device_inference.device_readiness` —
  :func:`~qml_project.device_inference.device_readiness.build_device_readiness_bundle`
  for Section 8.6 readiness / validation dicts handed to Sections 10–11.
- :mod:`qml_project.device_inference.device_cost` —
  :func:`~qml_project.device_inference.device_cost.build_device_cost_estimates_dataframe`
  for Section 10.2 per-winner circuit / shot budget tables.
- :mod:`qml_project.device_inference.result_sweep` —
  :func:`~qml_project.device_inference.result_sweep.run_device_inference_result_sweep`
  for Section 10.4 device submissions and workflow-cache result pickles.

Both pipelines refit on a small training budget (``DEVICE_TRAIN_SIZE = 50``
by default) so QSVM's support-vector count stays tractable on free-tier
shots and the device comparison is apples-to-apples with VQC.
"""

from __future__ import annotations

from .device_cost import (
    build_device_cost_estimates_dataframe,
    sum_device_cost_circuits_by_pipeline,
)
from .device_readiness import build_device_readiness_bundle
from .qsvm import (
    QSVMDevicePayload,
    build_qsvm_device_pubs,
    decode_qsvm_counts,
    refit_qsvm_for_device,
)
from .refit_sweep import run_device_refit_sweep_and_cache
from .result_sweep import (
    DeviceInferenceSectionBundle,
    load_disk_device_result_bundles,
    run_device_inference_result_sweep,
)
from .vqc import (
    VQCDevicePayload,
    build_vqc_device_pubs,
    decode_vqc_counts,
    refit_vqc_for_device,
)

__all__ = [
    "QSVMDevicePayload",
    "VQCDevicePayload",
    "build_device_cost_estimates_dataframe",
    "build_device_readiness_bundle",
    "build_qsvm_device_pubs",
    "build_vqc_device_pubs",
    "decode_qsvm_counts",
    "decode_vqc_counts",
    "DeviceInferenceSectionBundle",
    "load_disk_device_result_bundles",
    "refit_qsvm_for_device",
    "refit_vqc_for_device",
    "run_device_inference_result_sweep",
    "run_device_refit_sweep_and_cache",
    "sum_device_cost_circuits_by_pipeline",
]
