# NATIVEON: Wavefront Scheduler & Command Dispatch

> **Status:** CPU VERIFIED

---

## 1. Wave Dispatch Dynamics

The NATIVEON Scheduler dispatches workgroups across CUs/WGPs targeting maximum occupancy. Each workgroup consists of 1 to 16 Wavefronts (32 or 64 threads each).

---

## 2. Command Buffer Submission

Command buffers pack ISA instruction sequences into executable kernels. The NATIVEON execution engine executes programs deterministically sequentially or concurrently across simulated CUs.
