"""Encoding circuit builders.

Implements the three staged encodings used by the QML project:

1. **Angle encoding** — ``RY(h_i * pi / M)`` per heap; optional fourth qubit with
   ``RY(nim_sum * pi / M)`` when ``include_nim_sum`` is true (same scalar feature
   as amplitude, not L2-mixed).
2. **Amplitude encoding** — normalised heap amplitudes with optional Nim-sum
   coordinate in the same feature vector.
3. **Binary encoding** — heap bits plus optional fixed-width Nim-sum bit
   register; per-bit-position CZs on heap qubits mirror XOR structure.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from qiskit import QuantumCircuit

from qml_project.nim.game import NimState, nim_sum

EncodingName = Literal["angle", "amplitude", "binary"]
SymmetryMode = Literal["none", "canonical", "equivariant"]
ENCODING_CANDIDATES: tuple[EncodingName, ...] = ("angle", "amplitude", "binary")


def _canonicalise_state(state: NimState, *, symmetry: SymmetryMode) -> tuple[int, ...]:
    """Return a symmetry-adjusted state used before encoding."""
    arr = np.asarray(state, dtype=np.int32)
    if arr.ndim != 1:
        raise ValueError("state must be a 1D heap-size vector")
    if symmetry == "canonical":
        arr = np.sort(arr)
    return tuple(int(v) for v in arr)


def angle_parameters(
    state: NimState,
    *,
    M: int = 7,
    include_nim_sum: bool = True,
    symmetry: SymmetryMode = "none",
) -> np.ndarray:
    """Map heaps to rotation angles ``theta_i = h_i * pi / M``.

    When ``include_nim_sum`` is true, append ``nim_sum(state) * pi / M`` so the
    classical feature set matches the optional Nim-sum channel used in
    :func:`amplitude_vector`.
    """
    state_sym = _canonicalise_state(state, symmetry=symmetry)
    angles = [float(h) * np.pi / float(M) for h in state_sym]
    if include_nim_sum:
        angles.append(float(nim_sum(state_sym)) * np.pi / float(M))
    return np.asarray(angles, dtype=np.float64)


def build_angle_encoding_circuit(
    state: NimState,
    *,
    M: int = 7,
    include_nim_sum: bool = True,
    symmetry: SymmetryMode = "none",
) -> QuantumCircuit:
    """Build angle encoding: 3 qubits, or 4 when ``include_nim_sum`` is true."""
    theta = angle_parameters(
        state, M=M, include_nim_sum=include_nim_sum, symmetry=symmetry
    )
    qc = QuantumCircuit(len(theta))
    for q, angle in enumerate(theta):
        qc.ry(float(angle), q)
    return qc


def amplitude_vector(
    state: NimState,
    *,
    M: int = 7,
    include_nim_sum: bool = True,
    symmetry: SymmetryMode = "none",
) -> np.ndarray:
    """Build a normalised amplitude vector for pilot experiments.

    The feature vector uses normalised heap sizes and (optionally) Nim-sum:
    ``[h1/M, h2/M, h3/M, nim_sum(state)/M]``. The vector is L2-normalised and
    has dimension 4, so the circuit uses 2 qubits.
    """
    state_sym = _canonicalise_state(state, symmetry=symmetry)
    vals = [float(v) / float(M) for v in state_sym]
    if include_nim_sum:
        vals.append(float(nim_sum(state_sym)) / float(M))
    vec = np.asarray(vals, dtype=np.float64)
    if vec.size == 0:
        raise ValueError("amplitude vector cannot be empty")
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        vec = np.zeros_like(vec)
        vec[0] = 1.0
        target_len = 1 << int(np.ceil(np.log2(vec.size)))
        if vec.size < target_len:
            vec = np.pad(vec, (0, target_len - vec.size))
        return vec
    vec = vec / norm
    target_len = 1 << int(np.ceil(np.log2(vec.size)))
    if vec.size < target_len:
        vec = np.pad(vec, (0, target_len - vec.size))
    return vec


def build_amplitude_encoding_circuit(
    state: NimState,
    *,
    M: int = 7,
    include_nim_sum: bool = True,
    symmetry: SymmetryMode = "none",
) -> QuantumCircuit:
    """Build amplitude-encoding pilot circuit (2 qubits for 4 amplitudes)."""
    vec = amplitude_vector(
        state,
        M=M,
        include_nim_sum=include_nim_sum,
        symmetry=symmetry,
    )
    n_qubits = int(np.log2(vec.size))
    if 2**n_qubits != vec.size:
        raise ValueError("Amplitude vector length must be a power of two")
    qc = QuantumCircuit(n_qubits)
    qc.initialize(vec, list(range(n_qubits)))
    return qc


def binary_bits(
    state: NimState,
    *,
    bits_per_heap: int = 3,
    symmetry: SymmetryMode = "none",
) -> np.ndarray:
    """Encode each heap as fixed-width little-endian bits."""
    state_sym = _canonicalise_state(state, symmetry=symmetry)
    bits: list[int] = []
    for heap in state_sym:
        if heap < 0:
            raise ValueError("heap sizes must be non-negative")
        if heap >= 2**bits_per_heap:
            raise ValueError(
                f"heap value {heap} exceeds representable range "
                f"[0, {2**bits_per_heap - 1}]"
            )
        for bit in range(bits_per_heap):
            bits.append((heap >> bit) & 1)
    return np.asarray(bits, dtype=np.int8)


def build_binary_encoding_circuit(
    state: NimState,
    *,
    bits_per_heap: int = 3,
    include_nim_sum: bool = True,
    symmetry: SymmetryMode = "none",
) -> QuantumCircuit:
    r"""Build binary encoding with Nim-sum-aligned bit-position entanglement.

    Qubits are ordered as:
    ``[h1_b0, ..., h3_b_{B-1}, ns_b0, ..., ns_b_{B-1}]`` with ``B=bits_per_heap``.

    The first ``3 * B`` qubits are heap bits (little-endian per heap). The
    trailing ``B`` qubits are a **fixed-width Nim-sum register** (same width as
    one heap, sufficient for XOR of three ``B``-bit values). When
    ``include_nim_sum`` is false they stay in :math:`|0\rangle`; when true they
    are set from the integer ``nim_sum`` in little-endian bit order.

    For each heap bit position, the three heap qubits are entangled with CZ
    gates to mirror the bitwise XOR structure in Nim-sum (independent of the
    explicit Nim-sum register).
    """
    n_heaps = len(_canonicalise_state(state, symmetry="none"))
    if n_heaps != 3:
        raise ValueError("binary encoding currently expects exactly 3 heaps")
    state_sym = _canonicalise_state(state, symmetry=symmetry)
    bits = binary_bits(state, bits_per_heap=bits_per_heap, symmetry=symmetry)
    B = bits_per_heap
    n_heap_q = n_heaps * B
    n_sum_q = B
    qc = QuantumCircuit(n_heap_q + n_sum_q)

    for q, bit in enumerate(bits):
        if int(bit) == 1:
            qc.x(q)

    ns = int(nim_sum(state_sym))
    if ns < 0:
        raise ValueError("nim_sum must be non-negative")
    if ns >= 2**B:
        raise ValueError(
            f"nim_sum={ns} does not fit in bits_per_heap={B} "
            "(increase bits_per_heap or reduce heap range)"
        )
    if include_nim_sum:
        base = n_heap_q
        for bit in range(B):
            if (ns >> bit) & 1:
                qc.x(base + bit)

    for bit_pos in range(B):
        q0 = bit_pos
        q1 = B + bit_pos
        q2 = 2 * B + bit_pos
        qc.cz(q0, q1)
        qc.cz(q1, q2)
        if symmetry == "equivariant":
            # Symmetric closure so each heap pair sees a CZ at this bit position.
            qc.cz(q0, q2)
    return qc


def build_encoding_circuit(
    encoding: EncodingName,
    state: NimState,
    *,
    M: int = 7,
    bits_per_heap: int = 3,
    iqp_reps: int = 2,
    include_nim_sum: bool = True,
    symmetry: SymmetryMode = "none",
) -> QuantumCircuit:
    """Dispatch helper to build any staged encoding circuit by name.

    ``iqp_reps`` is accepted for backward compatibility with older call sites;
    it is not used by angle, amplitude, or binary encodings.
    """
    _ = iqp_reps
    if encoding == "angle":
        return build_angle_encoding_circuit(
            state,
            M=M,
            include_nim_sum=include_nim_sum,
            symmetry=symmetry,
        )
    if encoding == "amplitude":
        return build_amplitude_encoding_circuit(
            state,
            M=M,
            include_nim_sum=include_nim_sum,
            symmetry=symmetry,
        )
    if encoding == "binary":
        return build_binary_encoding_circuit(
            state,
            bits_per_heap=bits_per_heap,
            include_nim_sum=include_nim_sum,
            symmetry=symmetry,
        )
    raise ValueError(
        f"unknown encoding {encoding!r}; expected one of {ENCODING_CANDIDATES}"
    )
