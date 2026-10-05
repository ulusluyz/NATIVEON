# NATIVEON: Benchmark Suite & Performance Profiling

> **Status:** CPU VERIFIED / BENCHMARK PROFILING

---

## 1. Simulated CPU Reference Benchmark Results

| Primitive | Iterations | Avg Latency (CPU Engine) | SIMD Ops / sec | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **Wave Associative Mixing (WAM)** | 10,000 | ~11.4 µs | ~2.80M ops/s | **CPU VERIFIED** |
| **Register-Resident State Update (RRSU)** | 10,000 | ~9.6 µs | ~3.31M ops/s | **CPU VERIFIED** |

*Note: Latency values reflect CPU wavefront simulator overhead. Hardware execution latencies on physical AMD GPUs (MI300X / RX 7900 XTX) require direct ROCm execution.*
