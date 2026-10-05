# NATIVEON: Checkpoint Binary Serialization Format

> **Status:** CPU VERIFIED Specification

---

## Binary Layout Format

| Offset (Bytes) | Type | Field Name | Description |
| :--- | :--- | :--- | :--- |
| `0x00 - 0x07` | `u8[8]` | Magic Header | ASCII string `NATIVEON` |
| `0x08 - 0x0B` | `u32` | Vocabulary Size | `vocab_size` (Little Endian) |
| `0x0C - 0x0F` | `u32` | Hidden Dimension | `hidden_dim` (Little Endian) |
| `0x10 - 0x13` | `u32` | Wave Mode | `32` for Wave32, `64` for Wave64 |
| `0x14 - ...` | `f32[]` | Token Embeddings | Float32 array `[vocab_size * hidden_dim]` |
| `...` | `f32[]` | Decay Parameters | Float32 array `[hidden_dim]` |
| `...` | `f32[]` | Input Weights | Float32 array `[hidden_dim]` |
| `...` | `f32[]` | Output Projections| Float32 array `[vocab_size * hidden_dim]` |
