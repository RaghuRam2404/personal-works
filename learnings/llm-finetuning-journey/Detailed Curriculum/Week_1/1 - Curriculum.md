# Week 1 — PyTorch Tensors, Autograd, and the Training Loop

## Learning Objectives

By the end of this week, you will be able to:

- Create, manipulate, and index PyTorch tensors confidently, including broadcasting rules.
- Explain what a computational graph is and how PyTorch builds one dynamically.
- Trace `loss.backward()` through a small network by hand, matching it to PyTorch's computed gradients.
- Write the canonical PyTorch training loop from memory without referring to documentation.
- Identify and fix three common training-loop bugs: missing `optimizer.zero_grad()`, forgetting `model.eval()` during validation, and shapes mismatches in the loss function.
- Re-implement Karpathy's `micrograd` engine to prove you understand autodiff from first principles.

---

## Concepts

### Tensors

A tensor is a multi-dimensional array with a fixed dtype living on a device (CPU, CUDA, or MPS). PyTorch tensors are the unit of computation: every input, every weight, every gradient is a tensor.

Key operations you must be fluent with:

```python
import torch

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # shape (2, 2)
x.shape        # torch.Size([2, 2])
x.dtype        # torch.float32
x.device       # device('cpu')

# Indexing
x[0]           # first row  -> tensor([1., 2.])
x[:, 1]        # second col -> tensor([2., 4.])

# Operations — all create new tensors
x + x
x @ x          # matrix multiply
x.sum(dim=0)   # sum over rows -> shape (2,)
x.unsqueeze(0) # shape (1, 2, 2)
x.view(4)      # shape (4,) — shares memory
x.reshape(4)   # shape (4,) — may copy
```

**Broadcasting** is PyTorch's rule for operating on tensors with different shapes by implicitly expanding dimensions of size 1. The rule: align shapes from the right; dimensions must be equal, one of them must be 1, or one must be absent. A mismatch here is the single most common source of silent bugs. Example: `(3, 1) + (1, 4)` produces `(3, 4)`.

**Device placement** matters. Tensors default to CPU. Move to GPU with `.to('cuda')` or `.to('mps')` (Apple Silicon). Operations between tensors on different devices raise an error immediately — which is actually helpful for debugging.

### Autograd and the Computational Graph

When you set `requires_grad=True` on a tensor, PyTorch records every operation applied to it in a dynamic computational graph. This graph is built from scratch on every forward pass, which is what makes PyTorch flexible compared to older static-graph frameworks.

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1   # y = x^2 + 3x + 1
y.backward()
print(x.grad)             # dy/dx = 2x + 3 = 7.0 at x=2
```

The `backward()` call traverses the graph from `y` back to all leaf tensors with `requires_grad=True`, accumulating gradients via the chain rule. The key insight: **gradients accumulate**. If you call `backward()` twice without zeroing, the gradients add up. This is almost never what you want.

**Leaf tensors** are those you create directly (e.g., model parameters). **Non-leaf tensors** are outputs of operations. You can only access `.grad` on leaf tensors by default.

**The chain rule** is the mathematical engine here. For a composed function `L(y(x))`, the gradient `dL/dx = dL/dy * dy/dx`. PyTorch automates this across arbitrary computation graphs.

For deeper intuition, `micrograd` (Karpathy's 150-line implementation) walks through this from scratch. Building it yourself is the single most important exercise this week.

### The Canonical Training Loop

Every supervised deep learning training run has the same five-step loop inside the batch loop:

```python
for batch in dataloader:
    x, y = batch
    optimizer.zero_grad()      # 1. Zero old gradients
    y_hat = model(x)           # 2. Forward pass
    loss = criterion(y_hat, y) # 3. Compute loss
    loss.backward()            # 4. Backward pass
    optimizer.step()           # 5. Update weights
```

Step 1 is where beginners most often forget — the result is gradient accumulation across batches, leading to exploding updates. Step 2 runs the model's `forward()` method. Step 3 computes a scalar loss (PyTorch's `backward()` requires a scalar starting point by default). Step 4 populates `.grad` on all parameters. Step 5 uses those gradients to update weights according to the optimizer rule.

The **validation loop** differs in two ways: you wrap it in `torch.no_grad()` (skips graph construction, saves memory and compute) and you call `model.eval()` beforehand (disables dropout and switches batch norm to use running statistics).

```python
model.eval()
with torch.no_grad():
    for batch in val_loader:
        x, y = batch
        y_hat = model(x)
        val_loss += criterion(y_hat, y).item()
model.train()  # switch back after validation
```

### Micrograd — Autodiff from First Principles

Karpathy's `micrograd` is a scalar-valued autograd engine in ~150 lines of Python. Every `Value` object stores a data scalar and a `_backward` function that knows how to propagate gradients through that specific operation. When you call `.backward()` on the final output, it topologically sorts the graph and calls each `_backward` in reverse order.

Implementing this yourself forces you to internalize that:
1. Gradients are just local derivatives chained together.
2. The graph is a tree of operations, not magic.
3. Every primitive (`+`, `*`, `exp`, `tanh`) needs its own backward rule.

After building `micrograd`, porting your old plain-Python neural network to PyTorch will feel mechanical — you are just swapping your scalar `Value` objects for PyTorch tensors.

---

## Connections

**Prior knowledge used:** Linear algebra (matrix multiply, chain rule from calculus), your existing 1–2 layer Python NN.

**What this week unlocks:** Everything. Without autograd fluency, Weeks 2–8 (and the entire course) are inaccessible. Week 2's batch norm, Week 5's optimizer comparisons, and Week 8's capstone all require a solid mental model of the gradient flow you build here.

---

## Common Misconceptions and Pitfalls

- **"I don't need to understand autograd — PyTorch handles it."** Wrong. When your loss goes to NaN, you will need to trace exactly where the gradient exploded. Black-box thinking will block you at that moment.
- **Forgetting `optimizer.zero_grad()`.** This is the single most common training bug. The gradients accumulate across batches, producing unpredictable weight updates.
- **`loss.item()` vs `loss`.** Always call `.item()` when logging — otherwise you hold a reference to the entire computation graph in memory, causing a memory leak across training steps.
- **`view` vs `reshape`.** `view` requires contiguous memory and shares it; `reshape` copies if needed. Use `reshape` when in doubt.
- **Tensor on wrong device.** If you move the model to MPS but forget to move the batch, you get a device mismatch error. Always move both model and data to the same device.

---

## Time Allocation (6–8 hours this week)

| Activity | Time |
|---|---|
| Read PyTorch basics tutorial (all 8 sections) | 1.5 h |
| Watch Karpathy micrograd video (2h25m) — code along | 2.5 h |
| Implement micrograd from scratch | 1 h |
| Port your old Python NN to PyTorch | 1 h |
| Write `journal.md`, commit to GitHub | 30 min |
| Read Karpathy's "Recipe for Training NNs" | 30 min |
