# NATIVEON: WaveState SIMD Network Architecture

> **Status:** CPU VERIFIED / MODEL SPECIFICATION

---

## 1. Architectural Motivation

Unlike Transformers which require $O(N^2)$ memory bandwidth and full VRAM matrix transfers for Softmax Attention, the **NATIVEON WaveState SIMD Network** is designed around AMD GPU SIMD lane dynamics:

1. **Hidden State ($h_t$):** Partitioned across SIMD lanes inside VGPRs.
2. **Intra-Wave Mixing (WAM):** Inter-lane communication via single-cycle `ds_bpermute_b32` operations.
3. **In-Register Recurrence (RRSU):** Fused multiply-add (`v_fma_f32`) updates hidden state $h_t$ directly in VGPRs.

---

## 2. Information Flow Diagram

```
Input Token ID (t)
        │
        ▼
Token Embedding Vector [hidden_dim]
        │
        ▼
Register-Resident State Update (RRSU)  <--- (Previous Hidden State h_{t-1} in VGPRs)
        │
        ▼
Wave Associative Mixing (WAM - ds_bpermute)
        │
        ▼
New Hidden State h_t (Persists in VGPRs)
        │
        ▼
Output Projection Matrix ---> Vocabulary Logits
```
