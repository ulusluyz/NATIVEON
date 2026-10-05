# NATIVEON: AMD GPU Hardware Research & Architectural Foundations

> **Status:** CPU VERIFIED / THEORETICAL ANALYSIS
> **Target Architectures:** AMD CDNA3 (gfx900/gfx940) & AMD RDNA3 (gfx1100)

---

## 1. Executive Summary

Traditional Deep Learning frameworks (PyTorch, TensorFlow, CUDA-centric engines) treat GPUs as giant matrix-multiplication co-processors executing heavy 2D GEMM kernels (`C = A * B + C`). This paradigm forces models into standard Dense-Matrix Transformer structures (Attention + MLP) backed by large contiguous VRAM tensors.

The **NATIVEON** initiative re-evaluates artificial intelligence from the physical hardware primitives of AMD GPUs. Rather than asking *"How do we run Transformers efficiently on AMD GPUs?"*, NATIVEON asks:

> **"If we start from the raw physical execution characteristics of AMD GPU hardware (Wavefront execution, SIMD lane shuffles, Dual-Issue VOPD, SGPR/VGPR register files, Local Data Share LDS), what native compute model and learning architecture naturally emerges?"**

---

## 2. AMD GPU Hardware Characteristics Analysis

### 2.1 Wavefront Execution Model (Wave32 vs Wave64)

AMD GPUs execute instructions in lockstep units called **Wavefronts**.
* **RDNA3 (gfx1100):** Native **Wave32** (32 work-items/lanes per wavefront) with optional Wave64 mode.
* **CDNA3 (gfx900/gfx940):** Native **Wave64** (64 work-items/lanes per wavefront).

#### Architectural Implications for AI:
1. **Intra-Wave Communication (`ds_bpermute` & `v_readlane`):**
   Lanes within the same wavefront can exchange data in a single clock cycle without accessing global VRAM or even LDS, using hardware permutation primitives (`ds_bpermute_b32`).
2. **Divergence Penalty:**
   Branch divergence within a wavefront causes masked execution via the `EXEC` mask register (`s_and_saveexec_b64`), disabling inactive lanes. AI operations must avoid conditional branching per work-item and rely on bitwise masking.

### 2.2 Register Hierarchy (SGPR vs VGPR)

* **Scalar Registers (SGPR):** Shared across all 32/64 lanes in a wavefront. Used for uniform variables, loop counters, base addresses, and global state broadcasts.
* **Vector Registers (VGPR):** Dedicated per lane. On RDNA3, each Compute Unit features up to 256 VGPRs per SIMD (1536 VGPRs total per WGP).
* **Accumulator / Arch VGPRs:** CDNA3 features dedicated accumulator registers (`agpr`) for Matrix Fused Multiply-Add (MFMA).

#### Key Insight:
**Register-Resident AI State:** Register read/write throughput is orders of magnitude faster and lower latency than LDS or VRAM. An AI state that resides directly in VGPRs across a wavefront can execute recurrent state updates without incurring memory bandwidth latency.

### 2.3 Local Data Share (LDS) vs Global Memory (VRAM)

* **LDS (32KB - 64KB per CU/WGP):** High-speed, low-latency shared scratchpad memory with 32 banking structures.
* **Global Memory (HBM3 / GDDR6):** High throughput (up to 5.3 TB/s on CDNA3), but incurs 200-400 cycle latency on cache misses.

#### Key Insight:
For high-throughput, latency-sensitive sequence modeling, operations must maximize operational intensity $O = \text{FLOPs} / \text{Bytes Transferred}$. Storing recurrent state in VGPR + LDS eliminates global memory latency during token step generation.

### 2.4 Instruction-Level Parallelism & Dual-Issue VOPD (RDNA3)

RDNA3 introduced **VOPD (Vector Dual-Issue Operand)** instructions. A single 64-bit VOPD instruction packs two independent ALU operations (e.g., `v_dual_fmac` + `v_dual_add`) that execute simultaneously in a single SIMD clock cycle.

---

## 3. Physical Compute Primitives to AI Primitive Mapping

| Hardware Characteristic | Physical Compute Operation | NATIVEON AI Primitive |
| :--- | :--- | :--- |
| **Intra-Wave Shuffle** | `ds_bpermute_b32` / `v_readlane_b32` | **Wave Associative Mixing (WAM)** |
| **Scalar Broadcast** | `s_mov_b32` + Vector ALU | **Global State Broadcast (GSB)** |
| **Vector Register Locality** | VGPR-resident register banks | **Register-Resident State Update (RRSU)** |
| **Bitwise Masking** | `v_bfi_b32` / `v_and_b32` / `s_and_b32` | **Quantized Bit Mask Activation (QBMA)** |
| **Dual-Issue VOPD** | Parallel SIMD Multiply-Add | **Fused Dual-Path Recurrent Step (FDPRS)** |

---

## 4. Hardware Verification & Status Categorization

To maintain complete transparency and integrity (as mandated by NATIVEON principles):

* **CPU VERIFIED:** Validated using deterministic C++/Rust CPU reference implementation or CPU AMDGPU ISA Simulator.
* **HARDWARE VALIDATION REQUIRED:** Features designed for physical AMD GPU hardware execution requiring `hipExec`, ROCm, or raw AMDGPU command buffer submission on bare-metal AMD GPUs (e.g., MI300X, RX 7900 XTX).
