# Week 3 — Assignment

## Setup Checklist

- [ ] Google Colab account active. Open a new notebook, switch runtime to T4 GPU (Runtime → Change runtime type → T4 GPU). Verify: `import torch; print(torch.cuda.is_available())` → `True`.
- [ ] `torchvision` available on Colab: `import torchvision` — pre-installed, should work.
- [ ] W&B set up on Colab: `!pip install wandb -q` and `wandb.login()` with your API key.
- [ ] Karpathy makemore Parts 4 and 5 queued.

---

## Task 1 — Backprop Ninja (Makemore Part 4)

**Goal:** Derive and implement the backward pass for every operation in the makemore MLP without using autograd.

**Requirements:**
- Watch [Making makemore Part 4: Becoming a Backprop Ninja](https://www.youtube.com/watch?v=q8SA3rM6ckI) fully. Code along.
- Save your implementation in `week_03/backprop_ninja.py`.
- Your code must implement manual backward passes for:
  - Embedding lookup gradient (`dC`)
  - Concatenation / reshape gradient
  - Linear layer gradient (`dW`, `db`, `dx`)
  - Batch norm forward + backward (full formula, not just `nn.BatchNorm1d`)
  - Tanh backward
  - Cross-entropy + softmax backward (the combined formula)
- Verify all manual gradients against PyTorch autograd using this pattern:
  ```python
  assert (dW - W.grad).abs().max() < 1e-4, "W gradient mismatch!"
  ```
- All assertions must pass.

**Deliverable:** `week_03/backprop_ninja.py` with all assertions green. Commit message will be part of `week-03-cnn-cifar10`.

---

## Task 2 — CNN on CIFAR-10

**Goal:** Train a CNN to ≥75% test accuracy on CIFAR-10.

**Requirements:**
- Work in Colab. Save notebook as `week_03/cifar10_cnn.ipynb`.
- Use `torchvision.datasets.CIFAR10` with these transforms for training:
  ```python
  transforms.Compose([
      transforms.RandomHorizontalFlip(),
      transforms.RandomCrop(32, padding=4),
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  ])
  ```
  (Normalization constants are the CIFAR-10 per-channel mean and std.)
- Define a CNN with at least 2 convolutional blocks (conv → BN → ReLU → pool) and a linear classifier head.
- For each conv layer, print the output shape before training (to verify your shape arithmetic).
- Train for at least 20 epochs.
- Log to W&B project `week-03-cnn-cifar10`: `train/loss`, `train/acc`, `val/loss`, `val/acc` per epoch.
- Report final test accuracy. Must be ≥ 75%.
- Generate and save a confusion matrix plot as `week_03/confusion_matrix.png`.

**Deliverable:** Colab notebook + W&B run link + confusion matrix PNG. GitHub commit `week-03-cnn-cifar10`.

**Hints:**
- Shape formula: `H_out = (H_in + 2*padding - kernel_size) / stride + 1`. Verify after each layer before training.
- Use `torch.optim.SGD(lr=0.1, momentum=0.9, weight_decay=5e-4)` as baseline. It outperforms Adam on CIFAR-10 with proper LR schedule.
- Add `torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)` to decay the LR over epochs.

---

## Task 3 — Conv Shape Quiz (Written)

**Goal:** Build reflexive fluency with conv output shapes.

**Requirements:**
- In `week_03/conv_shapes.md`, compute the output shapes for each of these conv layers applied to an input of `(1, 3, 32, 32)`:
  1. `nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)` → ?
  2. `nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)` → ?
  3. `nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=0)` → ?
  4. `nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0)` → ?
  5. Stacking (1) → MaxPool(2,2) → (1 again, now with 64 in_channels) → MaxPool(2,2) → Global Avg Pool → ?
- Show your arithmetic. Verify each answer by printing `output.shape` in Python.

**Deliverable:** `week_03/conv_shapes.md` with hand-derived answers + Python verification.

---

## Stretch Goals

- Watch makemore Part 5 (WaveNet) and implement the tree-of-fusers architecture. Compare validation loss to the flat MLP from Week 2. Does the hierarchical architecture perform better on SQL token sequences?
- Try replacing max pooling with strided convolutions (`stride=2` with no explicit pooling). Does it change accuracy? This is the style used in ResNets (which you will build in Phase 2).
- Implement a depthwise separable convolution (MobileNet style) and compare parameter count vs. accuracy against your standard CNN.
