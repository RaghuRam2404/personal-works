# Week 1 — Answers

---

**Q1. Answer: B**

`optimizer.zero_grad()` is missing. Without it, `.backward()` accumulates gradients into `.grad` on every step — they are never reset. By step 2, the gradients are twice what they should be; by step 100, they are enormous. Depending on the optimizer's adaptive behavior, this can either cause the loss to spike or to stop moving entirely (if the effective gradient direction becomes incoherent). This is the most common training bug beginners write.

**Why others are wrong:**
- A: A high LR would cause the loss to spike or oscillate, not stay flat.
- C: `loss.backward()` always goes after the loss computation — this is nonsensical.
- D: `model.train()` matters for dropout and batch norm layers, but a simple MLP without those would still produce decreasing loss even if this call is absent.

---

**Q2. Answer: B — `(5, 3, 4)`**

Broadcasting aligns from the right: `(5, 1, 4)` and `(3, 4)`. The `(3, 4)` becomes `(1, 3, 4)` (missing leading dim added). Then: dim 0: 5 vs 1 → 5. Dim 1: 1 vs 3 → 3. Dim 2: 4 vs 4 → 4. Result: `(5, 3, 4)`.

**Why others are wrong:**
- A: The shapes are compatible. The rule requires each dim to match, be 1, or be absent.
- C: `(5, 1, 4)` would only be the result if `b` had shape `(1, 4)` or `(4,)`.
- D: Broadcasting never multiplies dims; it expands them.

---

**Q3. Answer: B**

Every PyTorch tensor that is part of a computation graph holds a reference back to that graph (for potential backward calls). Appending `loss` (a tensor with `grad_fn`) keeps the entire graph alive in memory. After 1000 steps, you have 1000 computation graphs in RAM. The fix is `loss.item()`, which extracts the scalar Python float and releases the graph reference.

**Why others are wrong:**
- A: Batch size affects GPU memory for activations, not for this specific Python list accumulation pattern.
- C: `torch.no_grad()` is for the inference/validation loop; the training loop needs the graph.
- D: Missing `zero_grad` causes incorrect gradients, not memory leaks of this type.

---

**Q4. Answer: B**

In the chain rule, if a node `z` feeds into two downstream nodes, the total gradient through `z` is the sum of gradients from both paths. Using `=` instead of `+=` means the second path's gradient silently overwrites the first. PyTorch itself uses `+=` (gradient accumulation semantics) for exactly this reason. This bug produces silently wrong gradients with no error message.

**Why others are wrong:**
- A: No `AttributeError` — both `=` and `+=` are valid Python operations on floats.
- C: Python does not automatically sum gradients; you must explicitly accumulate.
- D: The graph topology is determined by `_prev`, not by the `_backward` logic.

---

**Q5. Answer: B**

`requires_grad=False` tells PyTorch not to build the computational graph for operations involving that tensor. During inference, you never call `.backward()`, so building the graph is pure waste: it allocates memory for intermediate activations and grad functions that will never be used. `torch.no_grad()` context manager achieves the same effect at block level and is the preferred pattern for inference loops.

**Why others are wrong:**
- A: `requires_grad` has no bearing on device placement.
- C: `requires_grad` controls gradient tracking, not in-place protection (that's `tensor.requires_grad_(True)` + in-place ops, which raises an error for a different reason).
- D: The optimizer skips parameters by design if you exclude them — `requires_grad=False` is indeed used for freezing, but the explanation here misses the main point.

---

**Q6. Answer: B**

`transpose` in PyTorch does not move data in memory. It returns a view with swapped strides. A `(3, 4)` tensor with default strides `(4, 1)` becomes after `.transpose(0, 1)` a `(4, 3)` tensor with strides `(1, 4)` — non-contiguous because elements of a row are not adjacent in memory. `view` requires contiguous storage. Fix: `.contiguous()` copies data into a new contiguous block, then `view` works. Alternatively, `reshape` handles this transparently (it copies if needed).

---

**Q7 (short answer — model answer):**

When `loss.backward()` is called, PyTorch traverses the computational graph in reverse topological order starting from the scalar `loss` node. At each node, it applies the chain rule: the gradient of the loss with respect to that node's inputs equals the gradient arriving at that node (from downstream) multiplied by the local partial derivative of the node's operation. For example, if node `h = x @ W`, the gradient arriving at `W` is `x.T @ grad_h` and at `x` it is `grad_h @ W.T`. This continues backward through every operation until all leaf tensors with `requires_grad=True` have their `.grad` attribute populated. The `.grad` on model parameters is what the optimizer reads in `optimizer.step()`.

---

**Q8 (short answer — model answer):**

Three hypotheses ranked by likelihood:
1. **A bad batch in the data** — a batch with extreme values or label noise causing an outsized gradient update. Most common cause of isolated spikes.
2. **Learning rate too high relative to gradient magnitude at that point** — the model has reached a curved region of the loss surface and a large step overshoots. This would produce a spike followed by partial recovery as subsequent steps pull it back.
3. **A numerical instability in the loss function** — e.g., `log(0)` from a logit near 0 in cross-entropy without epsilon clipping. Check `loss.isnan()` and `loss.isinf()` in your training loop.

---

**Q9 (short answer — model answer):**

Accessing `.data` bypasses autograd entirely — PyTorch does not record the in-place modification in the computation graph. If that tensor was an intermediate value needed for the backward pass (e.g., an activation), modifying it in-place corrupts the graph, producing silent incorrect gradients or a cryptic error on the next `.backward()` call. The safer alternative is `torch.no_grad()` for blocks where you intentionally skip gradient tracking, or `with torch.inference_mode()` for pure inference. For in-place parameter updates during custom optimization, use `param.data.add_(...)` only when you are certain no backward pass will reference that tensor — which is guaranteed after `optimizer.step()` in the standard loop.
