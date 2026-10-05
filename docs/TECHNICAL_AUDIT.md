# NATIVEON TECHNICAL AUDIT REPORT

> **Project:** NATIVEON (AMD GPU Native AI Architecture & Execution Engine)
> **Date:** February 2025
> **Audit Focus:** Technical Provenance, AMD Hardware Justifications, Primitive Verification & Simulation vs Physical GPU Disambiguation

---

## 1. Executive Summary & Audit Methodology

This report presents a thorough, technical audit of the **NATIVEON** codebase, compute primitives, and AI model architecture.

### Primary Rule of Audit:
**Separation of Simulation from Physical Proof.**
No CPU simulator execution time, instruction count, or simulated memory throughput is accepted as proof of physical AMD GPU performance. All claims requiring bare-metal execution on physical AMD GPUs (MI300X, RX 7900 XTX) are strictly tagged as **`HARDWARE VALIDATION REQUIRED`**.

---

## 2. In-Depth Primitive Audit

### 2.1 Wave Associative Mixing (WAM)

1. **Why was WAM created?**
   To enable cross-token context exchange within sequence steps without incurring the $O(N^2)$ memory bandwidth penalty of reading global VRAM attention matrices ($Q K^T V$).
2. **Which AMD Hardware Feature Motivated WAM?**
   The hardware `ds_bpermute_b32` (and `ds_permute_b32`) LDS-permute instruction. In AMD GPUs, `ds_bpermute_b32` allows work-items in a wavefront (32 lanes on RDNA3, 64 lanes on CDNA3) to exchange 32-bit VGPR values directly using the LDS crossbar routing network without writing to or reading from LDS RAM cells or VRAM.
3. **What Alternatives Were Evaluated?**
   - *Softmax Dense Attention:* Memory-bound; requires reading key/value tensors from VRAM ($O(N^2)$). Rejected due to VRAM bandwidth latency at low batch sizes.
   - *LDS Scratchpad Matrix Exchange:* Required writing to LDS, waiting for barrier synchronization (`s_barrier`), and reading back. Causes 2-way / 4-way LDS bank conflicts during unaligned strided reads.
4. **Is There Real Measurement Supporting WAM's Advantage?**
   - **CPU Simulator Verification:** `CPU VERIFIED` (Tested via `tests/test_compute_primitives.rs::test_wam_matches_cpu_reference`). Matches CPU reference logic exactly.
   - **Physical GPU Clock Cycle Proof:** **`HARDWARE VALIDATION REQUIRED`**. While AMD ISA manuals document `ds_bpermute_b32` as a low-latency intra-wave operation, exact latency and throughput on physical hardware must be measured via bare-metal kernel profiling (`s_memtime`).

---

### 2.2 Register-Resident State Update (RRSU)

1. **Why was RRSU created?**
   To eliminate sequence step VRAM roundtrips during autoregressive state updates ($h_t = f(h_{t-1}, x_t)$).
2. **Which AMD Hardware Feature Motivated RRSU?**
   The large per-SIMD Vector Register File (VGPR) capacity (up to 256 VGPRs per SIMD lane on RDNA3 / 512 Arch+Accumulator VGPRs on CDNA3) and Dual-Issue VOPD / `v_fma_f32` instructions.
3. **What Alternatives Were Evaluated?**
   - *VRAM State Checkpointing:* Storing hidden state $h_t$ in global VRAM after every token step $t$. Causes high latency overhead (200-400 GPU clock cycles per step).
   - *LDS Ring Buffer:* Storing state in LDS. Consumes LDS allocation space, reducing workgroup occupancy per CU.
4. **Is There Real Measurement Supporting RRSU's Advantage?**
   - **CPU Simulator Verification:** `CPU VERIFIED` (Tested via `tests/test_compute_primitives.rs::test_register_state_update`).
   - **Physical GPU Memory Latency Proof:** **`HARDWARE VALIDATION REQUIRED`**. Register access vs VRAM access ratio is well-established theoretically in AMD ISA specs, but exact physical speedup under memory pressure requires bare-metal validation.

---

### 2.3 Global State Broadcast (GSB)

1. **Why was GSB created?**
   To broadcast uniform global model parameters (decay coefficients, layer gains) across all SIMD lanes without duplicating scalar values into vector registers.
2. **Which AMD Hardware Feature Motivated GSB?**
   AMD GPU's decoupled Scalar ALU (SALU) & Scalar Register File (SGPR) executing alongside the Vector ALU (VALU). Scalar registers (`s0..s103`) can be read directly as operand sources by vector instructions (`v_fma_f32 v3, v1, s0, v2`) at single-lane register cost.
3. **What Alternatives Were Evaluated?**
   - *Vector Register Duplication:* Copying uniform scalars to VGPRs across all 32/64 lanes using `v_mov_b32`. Increases VGPR pressure, reducing wavefront occupancy per CU.
4. **Is There Real Measurement Supporting GSB's Advantage?**
   - **CPU Simulator Verification:** `CPU VERIFIED` (Verified in `src/emulator.rs` instruction execution engine).
   - **Physical GPU VGPR Occupancy Proof:** **`HARDWARE VALIDATION REQUIRED`**.

---

## 3. WaveState AI Architecture Audit

1. **Why was a Recurrent Structure Selected?**
   Standard Transformer architectures process tokens in parallel during training via casual attention masks, but incur an $O(N)$ memory/KV-cache growth bottleneck during inference. A recurrent state formulation allows constant $O(1)$ per-step memory footprint and keeps hidden states inside VGPR vector registers.
2. **Which AMD Hardware Features Derived This Choice?**
   - High VGPR register count per SIMD.
   - Low-latency intra-wave shuffle (`ds_bpermute_b32`).
   - Scalar/Vector ALU co-execution.
3. **What Other Model Structures Were Evaluated?**
   - *Dense Softmax Transformer:* Evaluated theoretically; rejected due to $O(N^2)$ VRAM bandwidth bottleneck.
   - *Pure Convolutional Network:* Requires global state buffers for long sequences.
4. **Did WaveState Emerge Naturally or Is It an Experimental Hypothesis?**
   - **STATUS:** **`EXPERIMENTAL HYPOTHESIS / CPU VERIFIED`**.
   - **Technical Reality:** WaveState was derived by asking how recurrence and sequence mixing can be formulated using only `ds_bpermute_b32` and `v_fma_f32`. While mathematically functional and capable of sequence loss reduction on the CPU simulator (`tests/test_model_train.rs`), its capacity, scaling laws, and long-context text generation quality compared to standard models remain an **experimental hypothesis** requiring large-scale training.

---

## 4. Software Architecture & Runtime Disambiguation

| Module | Simulated Aspect | Bare-Metal AMDGPU UAPI Reality | Gap / Missing Verification |
| :--- | :--- | :--- | :--- |
| `src/isa.rs` | Custom struct & byte encoder | Official LLVM AMDGPU ELF format (`amdgcn-amd-amdhsa`) | Direct binary generation for AMD HSA runtime |
| `src/emulator.rs` | CPU loop over Rust vectors | Hardware GPU Command Processor (CP) & Asynchronous Compute Queues | Bare-metal driver dispatch via `/dev/kfd` / `/dev/dri` |
| `src/compute.rs` | CPU scalar/vector arithmetic | Hardware SIMD execution on RDNA3/CDNA3 CUs | Physical clock cycle measurements via `s_memtime` |

---

## 5. Summary Audit Matrix (Mandatory Audit Table)

| Bileşen | Köken | AMD Donanım Gerekçesi | Kanıt | Mevcut Durum | Eksik Doğrulama |
|---|---|---|---|---|---|
| **ISA Backend** | AMD ISA Dokümantasyonu (gfx1100/gfx940) | Wave32/64 instruction encoding, SGPR/VGPR operands | Unit tests (`tests/test_wavefront.rs`) | **CPU VERIFIED** | **HARDWARE VALIDATION REQUIRED** (Gerçek `amdgcn` ELF üretimi ve KFD driver submission yapılmadı) |
| **Wavefront Simulator** | RDNA3 / CDNA3 execution model araştırması | Wave32/64 kilit adım yürütmesi, EXEC maskeleme, LDS scratchpad | Unit tests (`tests/test_wavefront.rs`) | **CPU VERIFIED** | **HARDWARE VALIDATION REQUIRED** (Fiziksel CU scheduling, cache latency ve occupancy ölçülmedi) |
| **WAM** | Intra-wave context karma ihtiyacı | `ds_bpermute_b32` tek döngülü LDS crossbar permütasyonu | Reference comparison (`tests/test_compute_primitives.rs`) | **CPU VERIFIED** | **HARDWARE VALIDATION REQUIRED** (Fiziksel GPU instruction timing & LDS conflict ölçümü yapılmadı) |
| **RRSU** | VRAM roundtrip latency'sini engelleme ihtiyacı | Geniş VGPR register bankı (SIMD başına 256/512 VGPR), `v_fma_f32` | State update tests (`tests/test_compute_primitives.rs`) | **CPU VERIFIED** | **HARDWARE VALIDATION REQUIRED** (Fiziksel VRAM bandwidth latency avantajı bare-metal GPU'da ölçülmedi) |
| **GSB** | Vector register baskısını azaltma ihtiyacı | Scalar ALU (SALU) & SGPR register file'ın VALU ile eşzamanlı çalışması | Simulator execution engine (`src/emulator.rs`) | **CPU VERIFIED** | **HARDWARE VALIDATION REQUIRED** (Fiziksel SGPR broadcast latency ve occupancy kazancı ölçülmedi) |
| **WaveState** | Hardware-native recurrent model arayışı | VGPR-resident state + WAM inter-lane mixing + scalar broadcast | Training loss test (`tests/test_model_train.rs`) | **EXPERIMENTAL HYPOTHESIS / CPU VERIFIED** | **HARDWARE VALIDATION REQUIRED** (Büyük ölçekli dil öğrenme kapasitesi ve fiziksel GPU throughput henüz kanıtlanmadı) |
| **Training** | In-register backprop ihtiyacı | SGPR/VGPR üzerinde türev ve parametre güncellemesi | Loss reduction test (`tests/test_model_train.rs`) | **CPU VERIFIED** | **HARDWARE VALIDATION REQUIRED** (Fiziksel GPU üzerinde multi-wave veya multi-CU paralel eğitim kanıtlanmadı) |
| **Inference** | $O(1)$ bellek maliyetli otoregresif üretim ihtiyacı | VGPR saklı state üzerinden sürekli sonraki token üretimi | Generation test (`tests/test_model_train.rs`) | **CPU VERIFIED** | **HARDWARE VALIDATION REQUIRED** (Gerçek AMD GPU üzerinde token/sn (TPS) hızı ölçülmedi) |
