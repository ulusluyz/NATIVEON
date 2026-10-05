# NATIVEON: Experimental Log & Unsuccessful Approaches

> **Status:** EXPERIMENTAL LOG

---

## Log of Experiments

### EXP-001: Softmax Attention over LDS Buffers
* **Hypothesis:** Storing $QK^T$ attention matrices in 64 KB Local Data Share (LDS) scratchpad memory would accelerate token context retrieval.
* **Result:** **UNSUCCESSFUL / REJECTED.** LDS bandwidth saturation occurred when context length exceeded 512 tokens due to 32-bank LDS access conflicts during softmax normalization across wavefronts.
* **Resolution:** Replaced Softmax Attention with Wave Associative Mixing (WAM) using `ds_bpermute_b32` directly in VGPRs.

### EXP-002: Uniform Matrix Multiplication in Wave32 Mode
* **Hypothesis:** Standard GEMM kernel execution on RDNA3 Wave32.
* **Result:** High VRAM bandwidth bottleneck at batch size 1 during sequence inference.
* **Resolution:** Shifted to Register-Resident State Updates (RRSU).
