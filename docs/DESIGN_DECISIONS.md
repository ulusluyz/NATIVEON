# NATIVEON: Design Decisions & Provenance Ledger

> **Status:** DOCUMENTATION / AUDIT LOG

Every key architectural, runtime, and model decision in NATIVEON is tracked in this document using the strict provenance template required by the NATIVEON specification.

---

## Decision Ledger

### DEC-001: Selection of Dual Wave32/Wave64 Execution Model

* **ORIGIN:** Analysis of AMD GPU ISA specifications across RDNA3 (gfx1100, native Wave32) and CDNA3 (gfx940, native Wave64).
* **HARDWARE MOTIVATION:** RDNA3 consumer/workstation GPUs execute 32-lane wavefronts by default, while CDNA datacenter GPUs execute 64-lane wavefronts. Hardcoding wavefront size would compromise performance on either architecture.
* **ALTERNATIVES:**
  1. Force Wave64 on all GPUs (penalizes RDNA3 wave occupancy).
  2. Force Wave32 on all GPUs (sub-optimal for CDNA3 MFMA units).
* **MEASUREMENT:** Simulator wave layout tests (`tests/test_wavefront.rs`).
* **STATUS:** CPU VERIFIED

---

### DEC-002: Wave Associative Mixing (WAM) instead of Dense Matrix Attention

* **ORIGIN:** Hardware benchmarking of `ds_bpermute_b32` single-cycle inter-lane shuffle vs dense VRAM $O(N^2)$ attention matrix reads.
* **HARDWARE MOTIVATION:** $O(N^2)$ attention requires reading key/value matrices from VRAM for every token. `ds_bpermute_b32` allows work-items in a wavefront to communicate directly across SIMD lanes in VGPRs without touching VRAM.
* **ALTERNATIVES:** Standard Softmax Attention (memory-bound at batch size 1).
* **MEASUREMENT:** CPU SIMD Reference Benchmark (`benches/compute_primitives_bench.rs`).
* **STATUS:** CPU VERIFIED (HARDWARE VALIDATION REQUIRED for hardware timing).

---

### DEC-003: Register-Resident State Loop (RRSU)

* **ORIGIN:** Hardware register latency analysis (1 cycle VGPR read vs 20-30 cycles LDS vs 200+ cycles VRAM).
* **HARDWARE MOTIVATION:** Keeping hidden state directly in VGPRs across sequence steps enables continuous sequence processing without global memory roundtrips.
* **ALTERNATIVES:** Storing state tensors in VRAM / LDS between sequence steps.
* **MEASUREMENT:** Memory throughput analysis on simulator.
* **STATUS:** CPU VERIFIED
