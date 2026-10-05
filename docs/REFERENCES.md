# NATIVEON: Research References & Bibliography

> **Status:** DOCUMENTATION

---

## 1. AMD Hardware Documentation & ISA Manuals

1. **AMD RDNA3 Instruction Set Architecture (ISA) Reference Guide**
   - Source: AMD Developer Documentation (gfx1100 family)
   - Relevance: Wave32 execution, VOPD instruction format, VGPR allocation, `ds_bpermute` lane permute behavior.

2. **AMD CDNA3 Instruction Set Architecture (ISA) Reference Guide**
   - Source: AMD Developer Documentation (gfx940 family / MI300)
   - Relevance: Wave64 execution, MFMA matrix acceleration, AGPR register banks, LDS banking architecture.

3. **AMDGPU Backend - LLVM Compiler Infrastructure**
   - Source: LLVM Compiler Infrastructure (`llvm/lib/Target/AMDGPU`)
   - Relevance: Instruction encoding/decoding, register constraints, relocation formats.

---

## 2. Low-Level GPU Compute & Execution Research

1. **"Dissecting the AMD RDNA Architecture"**
   - Analysis of wavefront dispatch, instruction cache, scalar vs vector scheduling, and memory latencies.

2. **"Efficient Inter-Lane Wavefront Communication in AMD GPUs"**
   - Analysis of `ds_bpermute_b32` latency vs LDS bank conflicts vs global memory transactions.

---

## 3. Hardware-Native AI & Non-Transformer Architectures

1. **State Space Models & Recurrent Memory Architectures**
   - Context: Linear state space dynamics ($h_t = A h_{t-1} + B x_t$) mapping directly to vector-register update loops ($v_{h} = v_{a} \cdot v_{h} + v_{b} \cdot v_{x}$).

2. **Quantized / Bitwise Neural Architectures**
   - Context: Utilizing vector bitwise instructions (`v_bfi_b32`, `v_and_b32`) for integer/ternary activation functions.
