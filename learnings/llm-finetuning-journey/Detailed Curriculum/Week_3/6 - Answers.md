# Week 3 — Answers

---

**Q1. Answer: A — `(1, 64, 14, 14)`**

Formula: `H_out = floor((H_in + 2*padding - kernel_size) / stride + 1)`.
`H_out = floor((28 + 2*2 - 5) / 2 + 1) = floor((27) / 2 + 1) = floor(13.5 + 1) = floor(14.5) = 14`.
Output: `(1, 64, 14, 14)`.

**Why others are wrong:**
- B: Would require same-size output, which needs `stride=1` and `padding=2` for a 5-kernel.
- C: `(12,12)` results from stride=2, padding=0: `(28-5)/2+1=12.5→12`. Padding=2 adds 4 to effective input.
- D: Would require different arithmetic — check by substitution.

---

**Q2. Answer: B**

A 20-point gap between train (85%) and test (65%) is classic overfitting. Batch norm reduces internal covariate shift but does not prevent memorization. Data augmentation increases the effective diversity of training examples, and dropout reduces co-adaptation of neurons. Both directly attack overfitting.

**Why others are wrong:**
- A: More filters increase model capacity and would worsen overfitting.
- C: Reducing LR to 1e-5 would just slow training — it does not address the train/test gap.
- D: Batch norm helps generalization; removing it would make things worse.

---

**Q3. Answer: C**

The combined softmax + cross-entropy gradient is one of the most important formulas in deep learning: `dL/d_logits[i][j] = (probs[i][j] - 1) if j == y[i] else probs[i][j]`. This simplifies to `probs` with `1` subtracted only at the correct class index, then divided by `N` because the loss is averaged over the batch. Derivation: `L = -log(probs[y])`, so `dL/d_logit[k] = probs[k] - (k == y)`.

---

**Q4. Answer: B**

WaveNet is an autoregressive model: it generates audio sample-by-sample, conditioning each prediction on all previous samples. If a convolution could see future samples (two-sided), the model would effectively cheat by using ground-truth future audio during training. At inference, no future samples exist, causing a severe train/inference mismatch. Causal masking (only looking left, not right) is the same principle that makes GPT's causal self-attention work.

---

**Q5. Answer: B — `(1, C, 8, 8)`**

`MaxPool2d(2, 2)` halves each spatial dimension: `32 → 16 → 8`. Two pooling layers: `32 / 2 / 2 = 8`. Channels `C` depend on your last conv block.

---

**Q6. Answer: B**

Consider a batch where token index 5 (`SELECT`) appears in positions 0, 1, and 3. The gradient with respect to the embedding for token 5 is the sum of gradients flowing through all three occurrences (chain rule: same parameter, multiple uses). If you use `dC[5] = dout[0] + dout[1] + dout[3]` sequentially with `=`, the second write overwrites the first. With scatter-add (`dC.scatter_add_(0, ix, dout)`), all contributions are correctly summed. PyTorch's `nn.Embedding` uses exactly this in its backward pass.

---

**Q7 (short answer — model answer):**

Parameter sharing means a single convolutional filter is applied at every spatial position. A 3×3 conv with 64 output channels has `3 * 3 * 3 * 64 = 1,728` parameters regardless of input size. In contrast, a fully-connected layer from a flattened `32×32×3 = 3,072` input to 64 outputs requires `3,072 * 64 = 196,608` parameters — 114× more. Beyond efficiency, parameter sharing enforces the inductive bias that the same feature detector (edge, curve) should be useful everywhere in the image, not just at position (i,j). This prior is exactly right for natural images where features recur across positions.

---

**Q8 (short answer — model answer):**

Naively stacking more layers in a plain CNN without residuals causes degradation because the vanishing gradient problem worsens: gradients flowing back through many ReLU layers shrink (values <= 1 multiplied repeatedly → near 0), and deeper layers update very slowly. The network cannot even learn the identity function easily in the deeper layers.

Architectural change available this week: global average pooling instead of flatten → dense, which reduces the parameter count in the classifier and acts as built-in spatial regularization. Training-procedure change: stronger data augmentation — add `RandomRotation(15)` and `ColorJitter` to the transform pipeline. These add effective variance to training examples without any architectural changes and typically yield 2–4% accuracy gains on CIFAR-10.

---

**Q9 (short answer — model answer):**

A standard 1D conv with kernel size 3 and `k` layers has a receptive field of `2k + 1`. With 10 layers, RF = 21 time steps. WaveNet with dilation schedule `1, 2, 4, 8, ..., 512` over 10 layers has RF = `2 * (1 + 2 + 4 + ... + 512) + 1 = 2 * 1023 + 1 = 2047` time steps. That is 97× larger with the same number of parameters and the same depth. The key is that dilated convolutions skip intermediate positions, creating exponentially expanding receptive fields per layer while keeping the parameter count constant (the kernel still has 3 elements, just at wider spacing).
