# NATIVEON: Bare-Metal Physical AMD GPU Validation Procedures

> **Status:** HARDWARE VALIDATION REQUIRED

---

## 1. Physical AMD GPU Hardware Requirements

Execution on bare-metal hardware requires an AMD GPU supporting ROCm / HIP driver runtime:
* **Datacenter:** AMD Instinct MI300X / MI300A / MI250X (CDNA3 / CDNA2 - `gfx940`/`gfx90a`)
* **Workstation / Gaming:** AMD Radeon RX 7900 XTX / RX 7900 XT / PRO W7900 (RDNA3 - `gfx1100`)

---

## 2. Bare-Metal Validation Procedure

When deploying NATIVEON onto physical AMD GPU hardware:

1. **Verify ROCm Driver Stack:**
   ```bash
   rocminfo
   rocm-smi
   ```

2. **Verify ISA Kernel Compilation:**
   Assemble raw AMDGPU ISA kernel sources using `clang` / `lld`:
   ```bash
   clang -target amdgcn-amd-amdhsa -mcpu=gfx1100 -c kernel.s -o kernel.o
   ```

3. **Execute Bare-Metal Validation Script:**
   ```bash
   cargo run --bin nativeon -- --validate-hardware
   ```

4. **Status Tag Transition:**
   Upon successful hardware run, update component status in documentation from `HARDWARE VALIDATION REQUIRED` to `GPU VERIFIED`.
