# NATIVEON: AMDGPU ISA Backend Specification

> **Status:** CPU VERIFIED / AMDGPU ISA SPECIFICATION

---

## 1. Supported Opcodes & Instruction Map

NATIVEON ISA Backend defines native instruction representations for:

* `v_add_f32`, `v_mul_f32`, `v_fma_f32`: Vector Floating-Point Arithmetic.
* `v_bfi_b32`, `v_and_b32`, `v_or_b32`, `v_xor_b32`: Vector Bitwise Bit-Field Operations.
* `ds_bpermute_b32`: Data Share Wave Permutation.
* `s_mov_b32`, `s_add_u32`, `s_and_b64`: Scalar Control & Loop Operations.

---

## 2. Register Directives

* `SGPR0 - SGPR103`: 104 Scalar Registers per Wavefront.
* `VGPR0 - VGPR255`: 256 Vector Registers per SIMD lane.
* `EXEC`: 64-bit Execution Mask (`s_and_saveexec_b64`).
