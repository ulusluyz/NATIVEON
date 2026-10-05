# NATIVEON: AMD GPU Architecture Breakdown

> **Status:** CPU VERIFIED / THEORETICAL ANALYSIS
> **Architectures:** AMD CDNA3 (gfx940 / MI300 series) & AMD RDNA3 (gfx1100 / RX 7000 series)

---

## 1. CDNA3 vs RDNA3 Architectural Comparison

| Feature | AMD CDNA3 (gfx940) | AMD RDNA3 (gfx1100) | NATIVEON Architectural Decision |
| :--- | :--- | :--- | :--- |
| **Primary Execution Unit** | Compute Unit (CU) | Workgroup Processor (WGP) / Dual CU | Dual execution model abstraction |
| **Wavefront Size** | Wave64 native | Wave32 native (Wave64 mode supported) | Wave32/Wave64 agnostic SIMD abstraction |
| **Vector Registers (VGPR)** | 512 VGPRs (Arch + Accumulator) | 256 VGPRs per SIMD | Maximize 64 VGPR footprint per lane for occupancy |
| **Matrix Hardware** | MFMA (Matrix Fused Multiply-Add) | WMMA (Wave Matrix Multiply-Accumulate) | Matrix Primitives abstracted over SIMD lanes |
| **LDS Capacity** | 64 KiB per CU | 128 KiB per WGP (64 KiB per CU) | 32 KiB LDS scratchpad allocation per wavefront |
| **Inter-Lane Permutation** | `ds_bpermute_b32` / `ds_permute_b32` | `ds_bpermute_b32` / Dual VOPD | Wave-shuffle based associative token mixing |

---

## 2. Low-Level Execution pipeline & Scheduler Model

```
                    ┌─────────────────────────┐
                    │    Host Command Buffer  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  NATIVEON GPU Dispatch  │
                    └────────────┬────────────┘
                                 │
                   ┌─────────────┴─────────────┐
                   │                           │
                   ▼                           ▼
        ┌─────────────────────┐     ┌─────────────────────┐
        │  Scalar Engine (SQ) │     │  Vector SIMD Engine │
        │  SGPR State & Loop  │     │  VGPR State & ALU   │
        └──────────┬──────────┘     └──────────┬──────────┘
                   │                           │
                   └─────────────┬─────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Intra-Wave Exchange   │
                    │   `ds_bpermute_b32`     │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  VGPR Recurrent State   │
                    └─────────────────────────┘
```

---

## 3. Instruction Encoding & Assembly Layout

NATIVEON targets the native AMDGPU ISA. Instruction formats utilized in NATIVEON low-level kernels include:

1. **VOP1 / VOP2 / VOP3:** Vector ALU operations (`v_add_f32`, `v_mul_f32`, `v_fma_f32`, `v_bfi_b32`).
2. **SOP1 / SOP2 / SOPK:** Scalar operations (`s_mov_b32`, `s_add_u32`, `s_and_b32`).
3. **DS (Data Share):** Local Data Share instructions (`ds_bpermute_b32`, `ds_read_b32`, `ds_write_b32`).
4. **FLAT / GLOBAL:** Vector memory operations for global VRAM accesses.
