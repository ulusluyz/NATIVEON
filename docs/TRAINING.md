# NATIVEON: Training Pipeline Specification

> **Status:** CPU VERIFIED

---

## 1. Learning Objective

NATIVEON trains on sequence token streams using Cross-Entropy Loss:

$$\mathcal{L} = - \sum_{t=1}^{T} \ln P(w_t \mid w_{<t})$$

---

## 2. In-Register Backpropagation & Parameter Update

Gradients are computed on output projection matrices and embedding tables using stochastic gradient descent (SGD) or wave-parallel momentum updates.
