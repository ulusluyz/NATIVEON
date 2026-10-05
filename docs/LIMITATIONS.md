# NATIVEON: Known Hardware & Theoretical Limitations

> **Status:** DOCUMENTATION

---

## 1. Physical Hardware & Environment Constraints

1. **Bare-Metal GPU Access:**
   The current verification environment executes on Intel Xeon CPU host architecture without physical AMD GPU hardware (`/dev/kfd` unavailable). Physical hardware timing and occupancy measurements are marked as `HARDWARE VALIDATION REQUIRED`.

2. **Divergent Wave Operations:**
   Wavefront execution efficiency drops if operations cause lane divergence (`EXEC` mask branching). NATIVEON mitigates this by restricting all token updates to bitwise masking and SIMD uniform vectors.

3. **LDS Bank Conflicts:**
   Accessing LDS memory buffers across 32 banks simultaneously requires stride alignment to avoid 2-way or 4-way bank conflict stalls.
