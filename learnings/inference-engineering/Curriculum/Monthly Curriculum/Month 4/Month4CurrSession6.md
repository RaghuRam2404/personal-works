[← Month 4 overview](<../Month4.md>) · [← Session 5](<Month4CurrSession5.md>) · [Session 7 →](<Month4CurrSession7.md>)

## Session 6 — Expanded manual backward

### Objective

Implement the complete backward pass one elementary operation at a time and compare every intermediate gradient with PyTorch autograd.

### Test harness

Retain gradients for intermediate tensors and define a comparison helper that reports:

- Exact-shape agreement.
- Maximum absolute difference.
- Relative difference or `allclose` result.
- Whether both tensors are finite.

Use float64 for a small gradient-check case if numerical noise obscures correctness, then repeat the working implementation in the normal training dtype.

### Work order

Differentiate backward through:

1. Mean negative selected log probabilities.
2. Logarithm.
3. Probability normalization.
4. Reciprocal and sum.
5. Exponentiation.
6. Max-shift used for numerical stability.
7. Second linear layer.
8. Tanh.
9. BatchNorm affine transform.
10. Normalization by inverse standard deviation.
11. Variance.
12. Centering by mean.
13. First linear layer.
14. Reshape/concatenation.
15. Embedding lookup.

Compare each manual gradient immediately instead of implementing everything and testing only at the end.

### Required correctness cases

- A fixed lecture-style mini-batch.
- A batch containing repeated character IDs.
- Batch size greater than 1.
- At least five random seeds.
- Non-symmetric parameter initialization so an incorrect formula is not accidentally masked.

### Assignment M4.6

Deliberately introduce each bug separately:

- Fail to sum a broadcast bias gradient.
- Use assignment rather than accumulation for embedding gradients.
- Drop the `1 / B` factor from a mean.
- Differentiate tanh using the pre-activation rather than the tanh output incorrectly.

For each bug, identify the first intermediate gradient that diverges from autograd and explain why.

### First-principles questions

1. Why compare every intermediate rather than only parameter gradients?
2. Why can a shape-correct gradient still be mathematically wrong?
3. Why do repeated token IDs create a scatter-add pattern?
4. Why does numerical stabilization with a max not alter softmax probabilities?
5. What role does the upstream gradient play at every node?

### Deliverable

- `makemore_deep/manual_backward.py` expanded backward.
- `tests/test_manual_backward.py`.
- Intermediate comparison table.
- Four failure analyses.

### Exit criteria

- Every intermediate and parameter gradient agrees with autograd within justified tolerance.
- You can locate a deliberately introduced bug from the first failed node.
