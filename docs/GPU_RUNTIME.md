# NATIVEON: Low-Level GPU Runtime Engine

> **Status:** CPU VERIFIED / SIMULATED EXECUTION
> **Target Subsystems:** AMDGPU ISA Encoder, Wavefront Execution Engine, SGPR/VGPR Allocation, LDS Simulator

---

## 1. Overview

The NATIVEON Low-Level GPU Runtime Engine bypasses high-level AI framework abstractions to interact directly with AMD GPU wavefront execution structures. It features:

1. **AMDGPU ISA Encoder/Decoder (`src/isa.rs`):** Structured instruction builder and binary encoder supporting VOP1, VOP2, VOP3, DS, and SOP instruction forms.
2. **Wavefront Execution Engine (`src/emulator.rs`):** Cycle-accurate simulator for Wave32 and Wave64 execution modes, managing 104 SGPRs per wavefront and up to 256 VGPRs per SIMD lane.
3. **Local Data Share (LDS) Scratchpad Engine:** Simulated 64 KiB LDS memory bank with fast read/write address indexing.
4. **Inter-Lane Communication Primitive (`ds_bpermute_b32`):** Hardware-accurate implementation of single-cycle wave shuffle allowing arbitrary 32-bit register permutations across work-items.

---

## 2. Low-Level Execution Pipeline

```
Instruction Program [Instruction]
        │
        ▼
Execution Engine Step Loop
        │
        ├── Instruction Decode & Register Selection
        ├── EXEC Mask Filtering (Bitmask per lane)
        ├── Execution on SGPR (Scalar Engine) or VGPR (SIMD Engine)
        └── Inter-lane Shuffle Buffer Swap (ds_bpermute_b32)
```

---

## 3. Verification & Test Suite

The ISA Backend and Emulator are validated through automated unit tests (`tests/test_wavefront.rs`):
* `test_wavefront_vadd`: Validates 32-lane parallel vector addition across VGPRs.
* `test_inter_lane_ds_bpermute`: Validates full 32-lane reverse permutation using `ds_bpermute_b32`.
