# NATIVEON: GPU Memory Architecture & Layout

> **Status:** CPU VERIFIED / THEORETICAL ANALYSIS

---

## 1. Memory Hierarchy Latencies & Bandwidth

| Memory Level | Capacity | Access Latency | Scope | Primary NATIVEON Usage |
| :--- | :--- | :--- | :--- | :--- |
| **VGPR / SGPR** | ~256 KB / SIMD | ~1 cycle | Lane / Wavefront | Active Recurrent State & Vector Math |
| **LDS Scratchpad** | 64 KB / CU | ~20-30 cycles | Workgroup | Inter-Wave Exchange & Shared Buffer |
| **L2 Cache** | 4-32 MB | ~60-120 cycles | GPU Global | Model Parameter Caching |
| **VRAM (HBM3/GDDR6)** | 16-192 GB | ~200-400 cycles | System | Model Checkpoint & Long-Term Memory |

---

## 2. Zero-VRAM Roundtrip Recurrent Loop

NATIVEON avoids writing recurrent state to VRAM on each sequence step. Recurrent hidden state vectors $h_t$ reside directly in VGPRs (`v0..v15`) across time steps $t=1..T$, keeping memory transactions restricted strictly to parameter reads.
