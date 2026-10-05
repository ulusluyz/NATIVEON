# NATIVEON: Autoregressive Inference Pipeline

> **Status:** CPU VERIFIED

---

## 1. Generation Pipeline

Given a prompt sequence $S = (t_1, t_2, \dots, t_k)$, the generator:
1. Feeds prompt tokens through the model to build initial VGPR recurrent hidden state $h_k$.
2. For step $i = k+1 \dots k+N$:
   - Computes output logits $L_i = \text{step}(t_{i-1}, h_{i-1})$.
   - Selects token $t_i = \text{argmax}(L_i)$ (or samples according to temperature $T$).
   - Appends $t_i$ to generated stream.
