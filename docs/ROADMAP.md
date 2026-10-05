# NATIVEON: Future Development Roadmap

> **Status:** ROADMAP

---

## Roadmap Phases

### Phase 1: Foundational Research & Simulator (CURRENT)
- [x] AMD GPU ISA and hardware research across CDNA3 and RDNA3.
- [x] CPU wavefront emulator & ISA instruction encoder (`src/isa.rs`, `src/emulator.rs`).
- [x] NATIVEON compute primitives (`WAM`, `RRSU`, `GSB`, `QBMA`).
- [x] WaveState SIMD Recurrent Network architecture (`src/model.rs`).
- [x] Training, Cross-Entropy loss, and Autoregressive inference engine (`src/train.rs`).
- [x] Binary checkpoint serialization (`src/train.rs`).
- [x] Benchmark suite (`src/benchmark.rs`).
- [x] Complete documentation suite (`docs/`).

### Phase 2: Bare-Metal Hardware Execution
- [ ] Direct HIP/ROCm kernel loading on physical MI300X / RX 7900 XTX hardware.
- [ ] Hardware instruction profiling and cycle counter measurements (`s_memtime`).
- [ ] Transition status tags from `HARDWARE VALIDATION REQUIRED` to `GPU VERIFIED`.

### Phase 3: Large-Scale Corpus Training & Multi-GPU Scale
- [ ] Inter-CU Wavefront dispatch optimization.
- [ ] Large-scale language modeling benchmark evaluations.
