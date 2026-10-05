# NATIVEON

> **Donanımın doğal hesaplama yapısından doğan yapay zekâ.**
> *(Artificial Intelligence Born from Hardware's Native Compute Structure)*

---

## What is NATIVEON?

**NATIVEON** is a ground-up AI architecture and custom runtime built specifically for **AMD GPU hardware** (RDNA3 gfx1100 and CDNA3 gfx940).

NATIVEON is **not** an attempt to port existing LLMs or PyTorch/CUDA codebases to ROCm/HIP. Instead, NATIVEON researches the physical compute characteristics of AMD GPUs—Wave32/Wave64 wavefront execution, intra-wave lane shuffles (`ds_bpermute_b32`), register bank layouts (SGPR/VGPR), and local data share (LDS)—and derives a native compute model and AI architecture optimized specifically for this hardware.

---

## Architectural Flow

```
AMD HARDWARE (RDNA3 / CDNA3)
        ↓
NATIVE EXECUTION CHARACTERISTICS (Wave32/64, VGPR/SGPR, LDS, VOPD)
        ↓
NATIVEON COMPUTE PRIMITIVES (WAM, RRSU, GSB, QBMA)
        ↓
NATIVEON AI ARCHITECTURE (WaveState SIMD Recurrent Network)
        ↓
TRAINING & LEARNING SYSTEM
        ↓
INFERENCE & CHECKPOINTING ENGINE
        ↓
INTELLIGENT TEXT OUTPUT
```

---

## Documentation Index

Detailed documentation for NATIVEON is located in the `docs/` directory:

1. [`docs/HARDWARE_RESEARCH.md`](docs/HARDWARE_RESEARCH.md) — AMD GPU hardware analysis, Wavefront dynamics, register structures.
2. [`docs/AMD_ARCHITECTURE.md`](docs/AMD_ARCHITECTURE.md) — Architectural breakdown of CDNA3 vs RDNA3.
3. [`docs/COMPUTE_MODEL.md`](docs/COMPUTE_MODEL.md) — NATIVEON compute primitives mathematical & hardware definition.
4. [`docs/GPU_RUNTIME.md`](docs/GPU_RUNTIME.md) — Low-level GPU runtime & CPU wavefront execution engine.
5. [`docs/MEMORY_ARCHITECTURE.md`](docs/MEMORY_ARCHITECTURE.md) — Register, LDS, and VRAM memory layout.
6. [`docs/ISA_BACKEND.md`](docs/ISA_BACKEND.md) — AMDGPU ISA encoder/decoder & assembly pipeline.
7. [`docs/SCHEDULER.md`](docs/SCHEDULER.md) — Wavefront scheduling & command buffer management.
8. [`docs/AI_ARCHITECTURE.md`](docs/AI_ARCHITECTURE.md) — The WaveState SIMD Recurrent Network architecture.
9. [`docs/MODEL_DESIGN.md`](docs/MODEL_DESIGN.md) — Token representation, state transitions, and loss functions.
10. [`docs/TRAINING.md`](docs/TRAINING.md) — Training pipeline, backpropagation through wave steps, optimizer.
11. [`docs/INFERENCE.md`](docs/INFERENCE.md) — Autoregressive text generation & sequence execution.
12. [`docs/CHECKPOINT_FORMAT.md`](docs/CHECKPOINT_FORMAT.md) — Binary checkpoint serialization format specification.
13. [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) — Performance benchmarks & execution profiling.
14. [`docs/HARDWARE_VALIDATION.md`](docs/HARDWARE_VALIDATION.md) — Physical AMD GPU verification procedures.
15. [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) — Known hardware, architectural, and theoretical constraints.
16. [`docs/DESIGN_DECISIONS.md`](docs/DESIGN_DECISIONS.md) — Provenance ledger tracking ORIGIN, MOTIVATION, and STATUS.
17. [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) — Experimental logs (including unsuccessful experiments).
18. [`docs/ROADMAP.md`](docs/ROADMAP.md) — Future development roadmap.
19. [`docs/REFERENCES.md`](docs/REFERENCES.md) — Research papers, ISA manuals, and technical references.

---

## Verification Status Definitions

* **THEORETICAL:** Conceptual design based on hardware documentation.
* **EXPERIMENTAL:** Implemented in code but undergoing experimental testing.
* **CPU VERIFIED:** Verified with exact bit/float accuracy using the CPU reference engine / AMD ISA simulator.
* **GPU VERIFIED:** Verified on physical AMD GPU hardware.
* **BENCHMARK VERIFIED:** Measured and profiled against baseline benchmarks.
* **HARDWARE VALIDATION REQUIRED:** Requires bare-metal AMD GPU hardware execution for confirmation.

---

## License

NATIVEON is released under modular licenses detailed in the repository:
* `LICENSE` / `LICENSE-RUNTIME` — MIT License for runtime and ISA tools.
* `LICENSE-MODEL` — MIT License for model architecture and weights.
* `LICENSE-DOCUMENTATION` — CC-BY-4.0 for technical documentation.
