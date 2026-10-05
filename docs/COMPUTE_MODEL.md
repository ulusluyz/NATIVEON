# NATIVEON: Compute Model Specification

> **Status:** CPU VERIFIED / SIMULATED GPU PRIMITIVES

---

## 1. NATIVEON Compute Primitives

NATIVEON replaces standard matrix-multiply primitives (`GEMM`) with four wavefront-native primitives:

### 1.1 Wave Associative Mixing (WAM)
* **Definition:** Inter-lane permutation and associate multiply-add within a single wavefront via `ds_bpermute_b32`.
* **Hardware Execution:** Executes in 2 GPU cycles without accessing VRAM or LDS.
* **Mathematical Formula:**
  $$y_i = x_{(i+1) \bmod N} \cdot x_i + x_i$$
* **CPU Reference Implementation:** `nativeon::compute::cpu_reference_wam()`

### 1.2 Register-Resident State Update (RRSU)
* **Definition:** Hidden state vector recurrence updated in-place inside VGPRs.
* **Hardware Execution:** Utilizes `v_fma_f32` with SGPR scalar broadcast parameters.
* **Mathematical Formula:**
  $$h_t = \gamma \cdot h_{t-1} + W_x \cdot x_t$$

### 1.3 Global State Broadcast (GSB)
* **Definition:** Broadcasts uniform global sequence parameters from an SGPR to all active VGPR lanes in 1 clock cycle using scalar operands.

### 1.4 Quantized Bit Mask Activation (QBMA)
* **Definition:** Fast non-linear activation using bitwise instructions (`v_bfi_b32`, `v_and_b32`).
