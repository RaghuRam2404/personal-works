# Week 3 — Convolutional Neural Networks, Backprop Ninja, and WaveNet

## Learning Objectives

By the end of this week, you will be able to:

- Compute the output shape of any convolutional layer from input shape, kernel size, stride, and padding — in your head, without code.
- Explain parameter sharing in CNNs and why it is the right inductive bias for spatially-structured data.
- Train a CNN on CIFAR-10 in PyTorch and achieve at least 75% test accuracy using Colab's T4 GPU.
- Derive the backpropagation rules for a convolution and a max-pooling layer by hand (the "backprop ninja" skill).
- Explain why WaveNet's dilated causal convolutions solved the long-range dependency problem for audio.
- Recognize the connection from WaveNet's hierarchical dilated convolutions to modern transformer architectures.

---

## Concepts

### Convolutions

A convolution applies a filter (kernel) of shape `(C_out, C_in, kH, kW)` to an input of shape `(N, C_in, H, W)` to produce output `(N, C_out, H_out, W_out)`. The key idea is **parameter sharing**: one kernel is slid across the entire spatial dimension, so `C_out * C_in * kH * kW` parameters cover an arbitrarily large input.

**Output spatial size formula:**
```
H_out = floor((H_in + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
```

For the common case (dilation=1):
```
H_out = (H_in + 2*padding - kernel_size) / stride + 1
```

Example: input `(1, 3, 32, 32)`, kernel `3×3`, stride 1, padding 1 → `H_out = (32 + 2 - 3)/1 + 1 = 32`. Padding=1 with kernel=3 preserves spatial size.

**Inductive biases CNNs encode:**
1. **Translation equivariance:** If a feature (e.g., an edge) shifts in the input, the activation map shifts by the same amount. The filter does not need to be re-learned for different positions.
2. **Locality:** Each output element only sees a local patch of the input (the receptive field). Features build from local to global across layers.

### Pooling and Receptive Field

**Max pooling** (`2×2`, stride 2) halves spatial dimensions while keeping the strongest feature. **Average pooling** is softer. **Global average pooling** collapses `(N, C, H, W)` to `(N, C)` — a powerful alternative to flattening before a classifier.

The **receptive field** of a neuron is the region of the original input it depends on. For a stack of `k` conv layers with kernel size 3 (stride 1, no padding): `RF = 2k + 1`. After 5 layers: RF = 11. Pooling or strided convolutions expand the RF without increasing parameters.

### Why CNNs Matter for an LLM Engineer

FlashAttention, the memory-efficient attention kernel behind modern LLMs, reuses CUDA tile-based computation patterns that descend directly from GPU-optimized convolution. Understanding conv shapes, memory layouts (NCHW vs. NHWC), and kernel fusion is the conceptual foundation.

More directly: 1D convolutions are the basis of WaveNet and are used in some efficient sequence models. Understanding depthwise convolutions and grouped convolutions (used in MobileNet, EfficientNet) gives you intuition for low-rank decompositions like LoRA (Phase 4).

### Backprop Ninja (Makemore Part 4)

Karpathy's "Backprop Ninja" session derives the backward pass for every operation in the makemore MLP by hand — not relying on PyTorch autograd. This is the most valuable exercise in Phase 1 for building intuition about gradient flow.

Key derivations to internalize:
- **BatchNorm backward:** Gradient through the normalization step involves both the gradient with respect to mean and variance, which depend on all examples in the batch. The full formula is non-trivial.
- **Cross-entropy + softmax backward:** Combined, `dL/d_logits = probs - one_hot(target)`, where `probs = softmax(logits)`. This is one of the most useful identities in deep learning.
- **Embedding backward:** The gradient w.r.t. embedding weights is a scatter-add: `dE[ix] += dout[i]` for each token `ix` in position `i`.

The "ninja" label refers to the skill of being able to write the backward pass for any operation without relying on autograd. You will use this skill when debugging gradient flow in large models and when implementing custom kernels (Phase 5–6).

### WaveNet and Dilated Causal Convolutions (Makemore Part 5)

WaveNet (van den Oord et al., 2016) generates raw audio waveforms autoregressively. The key architectural innovation is **dilated causal convolutions**: a 1D conv where each layer skips `d` steps, exponentially expanding the receptive field.

Dilation schedule: `d = 1, 2, 4, 8, 16, ...` — with 10 layers, RF = `2^10 - 1 = 1023` steps. For audio at 16kHz, that is ~64ms of context.

Karpathy's makemore Part 5 builds a WaveNet-style character-level LM: instead of a flat MLP reading the last N characters in parallel, a tree of 2-char fusers progressively merges character pairs into richer representations. This is conceptually a CNN over the token sequence.

The insight relevant to LLMs: modern LLMs are also hierarchical combiners, but they use attention to do the combining. WaveNet shows that hierarchy can be built convolutionally — cheaper but less flexible.

### CIFAR-10 and CNN Training

CIFAR-10: 60,000 `32×32` RGB images across 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). 50,000 train, 10,000 test. The standard first benchmark for CNNs.

A minimal architecture that achieves 75%+ test accuracy:

```
Conv(3→32, 3×3, pad=1) → BN → ReLU
Conv(32→32, 3×3, pad=1) → BN → ReLU → MaxPool(2×2)
Conv(32→64, 3×3, pad=1) → BN → ReLU
Conv(64→64, 3×3, pad=1) → BN → ReLU → MaxPool(2×2)
Flatten → Linear(64*8*8, 256) → ReLU → Dropout(0.3) → Linear(256, 10)
```

Parameters: ~500K. Training time: ~20 min on Colab T4 for 20 epochs. Use data augmentation (`RandomHorizontalFlip`, `RandomCrop`) for a significant free boost.

---

## Connections

**Builds on:** Week 2's batch norm, Kaiming init, training loop pattern.

**Unlocks:** Week 5's optimization experiments (you will re-run the CNN with better optimizers). The "backprop ninja" skill is foundational for understanding gradient checkpointing in Week 5 and LoRA gradient flow in Phase 4. WaveNet's dilated convolutions are a stepping stone to understanding attention patterns.

---

## Common Misconceptions and Pitfalls

- **"CNNs are only for images."** Wrong. 1D CNNs process sequences. Text models (TextCNN), speech models (WaveNet), and some time-series models all use 1D convolutions.
- **"Padding always means same output size."** Only if `padding = kernel_size // 2` with stride 1. Verify with the formula.
- **"Max pooling backprop passes gradient to the max."** Correct — but only to the max. If you are in the backprop ninja session, ensure you implement the argmax tracking correctly.
- **"More layers always = better."** Without residual connections (not introduced until Phase 2), very deep CNNs suffer from vanishing gradients. Stick to 4–8 layers for CIFAR-10.

---

## Time Allocation (6–8 hours this week)

| Activity | Time |
|---|---|
| Read CS231n ConvNets notes | 1 h |
| Watch makemore Part 4 (Backprop Ninja) — code along | 2 h |
| Watch makemore Part 5 (WaveNet) — code along | 2 h |
| Build CNN on CIFAR-10, train on Colab T4 | 2 h |
| Journal entry + GitHub commit | 30 min |
