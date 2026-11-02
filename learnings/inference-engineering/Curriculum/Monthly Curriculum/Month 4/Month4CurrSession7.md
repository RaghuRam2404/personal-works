[← Month 4 overview](<../Month4.md>) · [← Session 6](<Month4CurrSession6.md>) · [Session 8 →](<Month4CurrSession8.md>)

## Session 7 — Compact backward and training

### Objective

Collapse the expanded derivation into efficient layer-level gradients and use the manual backward pass in a real optimizer loop.

### Compact cross-entropy gradient

Derive and implement the combined gradient with respect to logits:

$$
\frac{\partial L}{\partial z}
=
\frac{p-y_{\text{one-hot}}}{B}
$$

Do not merely copy the expression. Explain how softmax and negative log-likelihood derivatives simplify into it.

### Compact BatchNorm gradient

Derive or reconstruct the layer-level gradient from the expanded graph. A valid compact form should account for:

- Direct dependence on normalized values.
- Dependence through batch mean.
- Dependence through batch variance.
- Per-feature summations across the batch.
- `gamma` scaling.

Check the compact result against both the expanded manual gradient and autograd.

### Manual training loop

Train the MLP without calling `loss.backward()`:

1. Forward pass.
2. Compute compact manual gradients.
3. Assign or store parameter gradients.
4. Update parameters under `torch.no_grad()`.
5. Clear temporary state.
6. Periodically compute train and development loss.

### Assignment M4.7

Compare three one-step updates from identical starting parameters:

- PyTorch autograd.
- Expanded manual backward.
- Compact manual backward.

Assert that resulting parameter tensors agree within tolerance.

Then run enough manual updates to show the loss falls and generated names remain valid.

### First-principles questions

1. Why is the compact formula faster than traversing every elementary node manually?
2. What information from the forward pass must be cached for backward?
3. Why is a fused backward implementation still just the chain rule?
4. What could go wrong if cached forward values are modified in place?
5. Why does successful loss reduction not replace gradient parity tests?

### Deliverable

- Compact cross-entropy backward.
- Compact BatchNorm backward.
- One-step parameter-update parity test.
- Manual-backward training curve.
- `outputs/manual_backward_results.md`.

### Exit criteria

- Compact and expanded gradients match.
- Manual training reduces loss.
- You can explain cross-entropy's compact gradient from first principles.
