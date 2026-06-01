"""Static prose for the multipage Nim QML demo.

Keeping titles, blurbs, and the FAQ here lets us edit narrative text
without touching logic. All strings are short by design — the demo pages
are one laptop-screen tall, so prose longer than two sentences is
usually a sign to cut rather than add.

**Math in Streamlit:** ``st.markdown`` renders **KaTeX**. Use inline
``$...$`` and display ``$$...$$`` (see ``ENCODING_EXPLANATIONS``). For a
single centred line without prose, pages may call ``st.latex(...)``.
"""

from __future__ import annotations

# Encodings exposed in the Streamlit demo (matches pilot comparisons; no IQP tab).
DEMO_ENCODINGS: tuple[str, ...] = ("angle", "amplitude", "binary")

# Long-form “how it works” for the Encoding learn page (Markdown + KaTeX).
ENCODING_EXPLANATIONS: dict[str, str] = {
    "angle": r"""
**Angle encoding (3 or 4 qubits)**

After the chosen **symmetry** step, heap $h_i$ maps to one rotation angle per
qubit (here $M=7$):

$$
\theta_i = \frac{h_i\,\pi}{M}
$$

With **Include Nim-sum** on (default), append
$\theta_4 = \mathrm{nim\_sum}(h_1,h_2,h_3)\,\pi/M$ and use **four** wires.
With it **off**, only $(\theta_1,\theta_2,\theta_3)$ appear — the same ablation
as Sections 05/06.

The circuit applies $\mathrm{RY}(\theta_i)$ on each wire. The register is a
**product state** (no entanglement from the encoding alone).
""".strip(),
    "amplitude": r"""
**Amplitude encoding (2 qubits)**

Let $(h_1,h_2,h_3)$ be heaps after symmetry. With **Include Nim-sum** on,
append $n = h_1 \oplus h_2 \oplus h_3$, stack four scalars, L2-normalise, then
zero-pad to length $4=2^2$:

$$
\tilde{\mathbf{v}} =
\left(\frac{h_1}{M},\frac{h_2}{M},\frac{h_3}{M},\frac{n}{M}\right)^{\!\top},
\qquad
\mathbf{v} = \frac{\tilde{\mathbf{v}}}{\|\tilde{\mathbf{v}}\|_2}
$$

With Nim-sum **off**, use only $(h_1,h_2,h_3)/M$, normalise, and pad to length
four on the same **two** qubits (one amplitude stays zero).

A **state-preparation** step loads $\mathbf{v}$ as computational-basis **amplitudes**
(QSVM overlap and device paths use Qiskit ``StatePreparation`` so inverses are
well-defined; ``Initialize`` bundles a reset and is avoided there).
Coherent loading of all components is why amplitude often wins at this budget.
""".strip(),
    "binary": r"""
**Binary encoding (9 heap qubits + 3 Nim-sum register qubits)**

Each heap $h_i\le 7$ uses **3 little-endian bits** on nine wires. With
**Include Nim-sum** on, three more qubits hold the Nim-sum bits (same width as
one heap); with it **off**, that register stays in $|0\\rangle$ but the
circuit width is unchanged — only the **X** pattern on the tail changes.

Apply **X** wherever a classical bit is $1$.

For each bit position $k\in\{0,1,2\}$, **CZ** gates entangle the three heap
qubits that carry bit $k$. **Equivariant** symmetry adds a **CZ** between the
first and third heap’s $k$-bit qubits so every heap pair is treated symmetrically.

This mirrors how Nim-sum **XORs** same-weight bits across heaps.
""".strip(),
}

# VQC learn page — ansatz & CZ strategy (must match ``build_circuit`` names).
ANSATZ_EXPLANATIONS: dict[str, str] = {
    "basic_block": r"""
**What it is:** On each qubit slot in every layer, the **basic block** applies
$\mathrm{RX}(\pi/2)$, then $\mathrm{RZ}(\phi)$ on the slot’s angle $\phi$
(either a **feature** parameter $x_k$ or a **trainable** weight $w_k$), then
another $\mathrm{RX}(\pi/2)$. That sandwich is a standard expressive
single-qubit “building block” before the layer’s **CZ** entanglement.

**Why use it:** Three rotations per slot give a flexible local unitary before
qubits interact; it matches common QML tutorial patterns and is the default
ansatz in many of this project’s Nim VQC sweeps when comparing expressivity
vs gate count.
""".strip(),
    "ry_rz": r"""
**What it is:** Each slot uses a lean stack: $\mathrm{RY}(\phi)$ followed by
$\mathrm{RZ}(\phi)$ with the **same** angle $\phi$ (feature or trainable)
on that qubit, then the layer’s **CZ** pairs.

**Why use it:** Fewer gates per slot than the basic block, so circuits are
shallower and cheaper to simulate or run on hardware at the cost of a
narrower single-qubit footprint. Useful in **ansatz comparisons** (ansatz
lottery) when you want to test whether extra local rotations buy real OOD
accuracy or mostly add noise.
""".strip(),
}

CZ_STRATEGY_EXPLANATIONS: dict[str, str] = {
    "linear": r"""
**What it is:** After each layer’s rotations, add **CZ** only on **adjacent**
qubit pairs $(0,1), (1,2), \ldots, (n_{\mathrm{q}}-2,\, n_{\mathrm{q}}-1)$ — a
chain along the register (here $n_{\mathrm{q}}$ is ``n_qubits``).

**Why use it:** Linear gate count in $n_{\mathrm{qubits}}$, topology matches
many physical layouts, and is easy to reason about. Trade-off: there is **no
direct** entanglement between distant qubits in one layer (only indirectly
through propagation across layers).
""".strip(),
    "all": r"""
**What it is:** After each layer’s rotations, apply **CZ** on **every**
distinct qubit pair $(i,j)$ with $i<j$ in that same layer.

**Why use it:** Maximally mixes the register within the layer (strong
multi-qubit correlations for a fixed depth budget). Trade-off: **gate count
scales like** $\mathcal{O}(n_{\mathrm{qubits}}^2)$ **per layer**, so cost
explodes as you add qubits — best reserved for small $n$ or ablation studies.
""".strip(),
    "random": r"""
**What it is:** After each layer’s rotations, pick **between 1 and**
$\min(3,\binom{n_{\mathrm{qubits}}}{2})$ **uniformly random** distinct pairs
and place **CZ** on those pairs only. Pair choices are **reproducible** given
``cz_seed`` (default **42** in ``build_circuit``).

**Why use it:** Explores **variable connectivity** without the full cost of
“all pairs”; helps design-space sweeps see whether lucky entanglement patterns
beat a fixed chain. Trade-off: each run’s geometry differs unless you fix the
seed and hyperparameters.
""".strip(),
}

# Long-form snippets imported by Learn pages (keep out of logic files).
VQC_READOUT_MARKDOWN = r"""
**Measurement → class.** The variational circuit ends with **measurement in
the computational (Z) basis** on every qubit, producing a **bitstring** each
shot. Finite-shot runs estimate **bitstring probabilities**; the exact
simulator used in training supplies the same object without sampling noise.

**Decision rule.** The project’s VQC payloads map bitstrings to the two Nim
labels via a fixed **bitstring-to-class table** learned during the sweep
(`class_map` in code). With **`decision_rule='argmax'`** (the usual setting),
we turn the two class probability masses into a single prediction by taking
the **argmax** over classes — i.e. whichever side (`winning` vs `losing`) has
larger total probability under the current $\boldsymbol{\theta}$ wins. Other
`decision_rule` / `observable` combinations exist for ablations but the
deployed checkpoints follow the notebook winner rows.
""".strip()

RESULTS_PAGE_CHART_CAPTION = (
    "One scatter: OOD balanced accuracy vs seconds (classical + Section 07 "
    "selection + IBM device). See `quantum_winners_summary` for rationale."
)

NOISE_PAGE_PLOT_FOOTNOTE = (
    "This chart is tier (3) only: each bar is balanced accuracy from a "
    "real IBM Runtime inference pass (Section 10 pickles). Tiers (1) clean "
    "statevector and (2) noisy Aer simulation are reported in the "
    "notebooks and workflow parquets (`vqc_workflow_df`, sweeps) — not redrawn "
    "here so the page stays a single glance at hardware degradation."
)

# Long-form ladder for the classical parity-features learn page (Markdown + KaTeX).
PARITY_FEATURES_MARKDOWN: str = r"""
**What you are looking at**

Classical baselines do not read raw heap counts alone. This page fixes a board
$(h_1,h_2,h_3)$ with $M=7$ and lists the **exact numeric vector** each
ablation feature set feeds to sklearn — same order as ``prepare_features``.

**Raw** — normalise heaps: $h_i/M$.

**Heap parities** — append $h_i \bmod 2$ (parity of each heap).

**Pairwise XOR** — append $(h_i \oplus h_j)/M$ for each pair $i<j$ (order matches the code: $(1,2),(1,3),(2,3)$).

**Bit parities** — for each bit position $b$ up to $\lceil \log_2(M+1)\rceil - 1$, XOR the $b$-th bit of $h_1$, $h_2$, and $h_3$. That value is bit $b$ of the Nim-sum $h_1 \oplus h_2 \oplus h_3$, stored as a float in $\{0,1\}$.

**``parity``** — concatenate **raw + heap parities + pairwise XOR + bit parities** (twelve numbers). Smaller sets stop after the corresponding block.
""".strip()

PAGE_TITLES: dict[str, str] = {
    "problem": "The problem: Nim",
    "data": "Data and labels",
    "parity_features": "Classical features",
    "encoding": "Input encoding",
    "vqc": "VQC architecture",
    "qsvm": "QSVM architecture",
    "classical": "Classical baselines",
    "training": "Training",
    "noise": "Noise and real device",
    "results": "Results",
    "faq": "FAQ",
}

PAGE_BLURBS: dict[str, str] = {
    "problem": (
        "Normal-play **Nim** with $k=3$ heaps and maximum heap size $M$. "
        "Whoever takes the last stone wins. A position is **winning** for "
        "the player to move if the **Nim-sum** is non-zero:\n\n"
        "$$h_1 \\oplus h_2 \\oplus h_3 \\neq 0$$\n\n"
        "Heap order does not matter ($S_3$ symmetry), which we exploit in "
        "both the encoding and the baselines."
    ),
    "data": (
        "Training data enumerates every reachable Nim state with heaps "
        "$\\le M_{\\mathrm{train}}=5$. The **OOD test set** uses states with "
        "at least one heap in $\\{6,7\\}$. The class is imbalanced "
        "($\\approx 12\\%$ losing), so we report **balanced accuracy** and "
        "**MCC** rather than raw accuracy. **Sample-efficiency curves** "
        "refit VQC/QSVM on nested subsets of the *same* OOD training pool "
        "($n \\in \\{25,50,100,150\\}$ plus the full train pool) so every point "
        "is comparable."
    ),
    "parity_features": (
        "Sklearn baselines use **hand-engineered** vectors built from heap sizes "
        "and Nim-sum structure — not the quantum encodings on the next page. "
        "Pick heaps below; each block shows the **components and values** for "
        "one ablation feature set (same names as the classical sweep)."
    ),
    "encoding": (
        "Three input encodings — **angle**, **amplitude**, **binary** — each "
        "with three **symmetry** modes and an **Include Nim-sum** toggle. "
        "Sections 05/06 cross **every encoding × Nim-sum on/off** for both VQC "
        "and QSVM (plus ansatz / depth / $C$, etc.). Toggle below to see how "
        "feature dimension and circuits change."
    ),
    "vqc": (
        "The VQC alternates **feature layers** (data-dependent angles) "
        "with **parameter layers** (trainable $\\boldsymbol{\\theta}$). "
        "Sweeps use the same **encoding × `include_nim_sum`** grid as QSVM "
        "(angle: 3 vs 4 features; amplitude: always four amplitudes on two "
        "qubits; binary: twelve angle slots for heap + Nim-sum bits). "
        "Apply a **Section 05 preset** to match those widths, then pick **ansatz** "
        "and **CZ strategy**. Measurement → bitstrings → **argmax** class rule."
    ),
    "qsvm": (
        "The QSVM uses a **fixed** quantum kernel — squared state overlap "
        "between encodings $x$ and $x'$:\n\n"
        "$$k(x,x') = \\bigl|\\langle\\psi(x)|\\psi(x')\\rangle\\bigr|^2$$\n\n"
        "Train an SVM on the precomputed $N\\times N$ kernel matrix; at "
        "inference you only need kernel values against **support vectors**. "
        "**Kernel preview:** pick encoding, symmetry, **Include Nim-sum**, and "
        "$N$ on a fixed shuffle of all legal `k=3, M=7` boards (exact "
        "statevector kernel; sweeps also vary $C$ and shot vs exact)."
    ),
    "classical": (
        "Three off-the-shelf sklearn classifiers — Logistic Regression, "
        "SVM (RBF), and Random Forest — each paired with five feature "
        "sets ranging from raw normalised heaps to engineered Nim-sum "
        "bit-parities. All use `class_weight='balanced'`."
    ),
    "training": (
        "Training logs live under the **`nim.<pipeline>.<stage>`** MLflow "
        "namespace (see `qml_project.training.experiment_namespace`). VQC: "
        "**COBYLA** on class-weighted softmax NLL over measured bitstring "
        "distributions. QSVM: **precomputed kernel** `SVC` on the exact "
        "statevector Gram matrix (shot kernels logged separately in sweeps). "
        "Classical: sklearn with `class_weight='balanced'`. Charts use cached "
        "parquets from Sections 05–07."
    ),
    "noise": (
        "The quantum pipelines are validated on three tiers: (1) exact "
        "**statevector** simulation, (2) a noisy Aer **fake-Brisbane** "
        "backend, and (3) **real IBM device inference** on a small test "
        "shard. Training always happens classically; only inference runs "
        "on hardware. **The bar chart on this page shows tier (3) only** "
        "(IBM Runtime); tiers (1)-(2) are in the notebooks and cached sweep "
        "tables."
    ),
    "results": "",
    "faq": "",
}


FAQ: list[tuple[str, str]] = [
    (
        "Why Nim?",
        "Nim is deterministic, compact (511 states for `k=3, M=7`), and "
        "has a **closed-form optimal strategy** based on the Nim-sum. "
        "That means we have ground-truth labels without any labelling "
        "noise, so any gap between a model and perfect play is due to "
        "model capacity, not label ambiguity.",
    ),
    (
        "Why only 3 heaps?",
        "`k = 3` keeps the circuits manageable (angle: **3 or 4** qubits "
        "depending on `include_nim_sum`; binary: **12** qubits for heap + "
        "Nim-sum register bits). The code supports arbitrary `k`, so scaling "
        "up is a configuration change, not a rewrite.",
    ),
    (
        "What does each class mean?",
        "Class 1 (`winning`) means the player **to move** can force a "
        "win with perfect play (Nim-sum ≠ 0). Class 0 (`losing`) means "
        "every move leads to a winning position for the opponent "
        "(Nim-sum = 0). The optimal move zeroes the Nim-sum.",
    ),
    (
        "Why balanced accuracy, not accuracy?",
        "Only ≈ 12% of states are losing, so a constant-`winning` "
        "predictor already gets ~ 88% raw accuracy. **Balanced accuracy** "
        "(mean per-class recall) is the headline ranking metric; see also "
        "the FAQ on **MCC**. Training uses `class_weight='balanced'` to "
        "offset the imbalance.",
    ),
    (
        "What is MCC and why show it?",
        "**Matthews correlation coefficient** summarises all four cells of "
        "the confusion matrix into one number in $[-1,1]$ (+1 is perfect, 0 "
        "is chance for balanced problems). It is stricter than raw accuracy "
        "on skewed labels and complements balanced accuracy: we log both in "
        "workflow parquets so sweeps can spot models that look good on BA "
        "but confuse the minority class.",
    ),
    (
        "How does the VQC turn measurements into a class?",
        "After the variational layers, every qubit is measured in the **Z** "
        "basis, yielding a **bitstring**. The pipeline converts empirical "
        "(or exact) **bitstring probabilities** into class probabilities "
        "using a fixed **bitstring→class map**, then applies the trained "
        "**`decision_rule`** — typically **`argmax`** over the `winning` vs "
        "`losing` probability totals. The Play tab’s circuit caption echoes "
        "the payload’s `observable` / `decision_rule` fields from MLflow.",
    ),
    (
        "What do train sizes n = 25, 50, … mean?",
        "They are **sample-efficiency** checkpoints: the same OOD train/test "
        "split is used, but quantum models are refit on **nested subsets** of "
        "the train pool (stratified by label) with $n$ states. Sections 05/06 use "
        "$n\\in\\{25,50,100,150\\}$ plus the **full** training pool so curves "
        "stay comparable across seeds.",
    ),
    (
        "What is **`include_nim_sum`** in the quantum encodings?",
        "It is an **ablation flag** logged on every VQC and QSVM run. **On:** "
        "angle gains a fourth data angle; amplitude’s normalised vector includes "
        "the Nim-sum scalar; binary **X**-loads the Nim-sum register from "
        "$h_1\\oplus h_2\\oplus h_3$. **Off:** angle uses three heap angles only; "
        "amplitude drops the Nim-sum component (one amplitude stays zero); "
        "binary leaves the Nim-sum register in $|0\\rangle$. Same encodings, "
        "one fewer explicit structural channel — we sweep both to see whether "
        "Nim-sum structure should live in the encoding or can be learned without it.",
    ),
    (
        "How was the winning quantum configuration chosen?",
        "`quantum_winners_summary.parquet` ranks candidates **within each "
        "pipeline** on mean **OOD balanced accuracy**, uses **std** across "
        "seeds as a stability tie-break, and treats **training wall-clock** "
        "as a cost axis (Pareto-style narrative). The `rationale` string on "
        "each row records that decision in plain language; the overall flag "
        "marks which pipeline heads the final three-way story.",
    ),
    (
        "Which QSVM hyperparameters are logged?",
        "Besides **encoding** and **symmetry**, sweeps log **`c_svc`** ($C$), "
        "**`estimator_mode`** (here `exact_statevector`), **`kernel_backend`** "
        "(`manual` vs fidelity helpers), **`bits_per_heap`**, **`iqp_reps`** "
        "(unused for the three encodings), **`include_nim_sum`**, **`train_size`**, "
        "**`seed`**, and **`stage`** "
        "into MLflow / `qsvm_workflow_df.parquet`. The Architecture page "
        "kernel preview uses the same exact-kernel defaults.",
    ),
    (
        "What are the ansatz lottery and barren plateaus?",
        "**Ansatz lottery:** small circuit changes (different CZ wiring or "
        "per-qubit blocks) can help or hurt accuracy in ways that are hard "
        "to predict a priori — sweeps exist to map that luck. **Barren "
        "plateaus** refer to exponentially flat gradients in deep "
        "parameterised circuits; this project keeps depth modest and monitors "
        "training loss curves to catch stalled optimisation early.",
    ),
    (
        "What about training cost and IBM queues?",
        "**Training** always ran locally (or cluster CPU) with cached wall "
        "times logged to MLflow — see the Training page. **IBM Quantum** jobs "
        "were reserved for **Section 10 inference**: short pub batches, queue "
        "waits dominated wall-clock, which is why we never attempted "
        "hardware training. Device pickles store the measured balanced "
        "accuracy plus runtime metadata from Runtime.",
    ),
    (
        "Why amplitude encoding?",
        "The pilot sweep compared **angle**, **amplitude**, and **binary** "
        "encodings on OOD balanced accuracy at a fixed qubit budget. "
        "Amplitude won for both VQC and QSVM: it loads "
        "$\\mathbf{v}\\propto(h_1/M,h_2/M,h_3/M,(h_1\\oplus h_2\\oplus h_3)/M)$ "
        "as one normalised amplitude vector on **two qubits**, so all four "
        "features are seen **coherently** with the shallowest circuits among "
        "the three.",
    ),
    (
        "Why `S_3` symmetry?",
        "Heap order carries no information in Nim. Ignoring that wastes "
        "model capacity on trivially equivalent states. We offer three "
        "symmetry modes — *none*, *canonical* (sort ascending), and "
        "*equivariant* (symmetry-aware circuit) — and log the choice per "
        "run in MLflow.",
    ),
    (
        "Why a fixed QSVM kernel, not a trained one?",
        "Trainable kernels were out of scope for this project. A fixed kernel "
        "$k(x,x')=|\\langle\\psi(x)|\\psi(x')\\rangle|^2$ plus an SVM gives a "
        "convex, globally optimal classifier per kernel choice — a clean "
        "counterpart to the non-convex VQC loss landscape.",
    ),
    (
        "Why only inference on a real device?",
        "Real-hardware queue times make training infeasible for this "
        "project. Every model is trained in simulation (statevector or "
        "Aer) and then validated by submitting the test-set inferences "
        "to an IBM backend. The device-inference results live in "
        "`notebooks/.workflow_cache/*_device_result_*.pkl`.",
    ),
    (
        "Which V2 primitives do you use?",
        "Project code uses `StatevectorSampler` + `SamplerPub` + "
        "`BindingsArray` everywhere, except in the cloud-inference "
        "notebook, which uses `qiskit_ibm_runtime.SamplerV2` on the "
        "IBM backend. Quantum kernels use exact statevector overlap "
        "(an explicit analytic baseline carve-out).",
    ),
    (
        "Where do the models come from?",
        "Trained checkpoints live in "
        "`notebooks/.workflow_cache/{vqc,qsvm}_device_payload_n*.pkl`. "
        "They were produced by the notebook pre-device / device sections "
        "and contain everything needed to re-run inference — circuits, "
        "trained parameters, support vectors, and metadata.",
    ),
]
