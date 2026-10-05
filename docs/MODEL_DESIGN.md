# NATIVEON: Model Design Specification

> **Status:** CPU VERIFIED

---

## 1. Parameters & Configuration

* `vocab_size`: Total vocabulary size (e.g. 256 for byte-level modeling).
* `hidden_dim`: Hidden dimension size (must be a multiple of wavefront size: 32 or 64).
* `wave_mode`: `Wave32` (RDNA3) or `Wave64` (CDNA3).

---

## 2. Parameter Tensors

1. `token_embedding`: Tensor of size `[vocab_size, hidden_dim]`.
2. `decay_params`: Decay coefficient vector of size `[hidden_dim]`.
3. `input_weights`: Input weight scalar vector of size `[hidden_dim]`.
4. `output_proj`: Projection matrix of size `[vocab_size, hidden_dim]`.
