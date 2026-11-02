[← Month 4 overview](<../Month4.md>) · [← Session 8](<Month4CurrSession8.md>) · [Session 10 →](<Month4CurrSession10.md>)

## Session 9 — Build the hierarchy

### Objective

Implement and train the WaveNet-style character model while preserving explicit shape reasoning.

### Required layers

Implement or finalize:

```python
class Linear
class BatchNorm1d
class Tanh
class Embedding
class FlattenConsecutive
class Sequential
```

`Linear` should operate on the final dimension of rank-2 or rank-3 input. `FlattenConsecutive(n)` should group adjacent positions and merge `n` neighboring feature vectors.

### Work order

1. Change the context length from 3 to a power-of-two value such as 8.
2. Verify the flat wider-context MLP first.
3. Implement `FlattenConsecutive(2)`.
4. Test it on a small numbered tensor whose grouping can be inspected manually.
5. Stack hierarchical blocks.
6. Verify each intermediate shape.
7. Ensure BatchNorm handles rank-3 input correctly.
8. Train and record train/development loss.
9. Generate names using the final model.
10. Compare parameter count and loss with the flat MLP.

### Required tests

- Correct grouping for `FlattenConsecutive`.
- Error when the time dimension is not divisible by the group size.
- Rank-2 behavior when the time axis collapses to 1.
- Linear transforms only the final dimension.
- BatchNorm preserves shape.
- No training/evaluation running-stat updates in evaluation mode.
- Final logits shape `[B, V]`.

### Assignment M4.9

Produce a comparison table:

| Model | Context length | Parameters | Aggregation depth | Train NLL | Dev NLL | Notes |
|---|---:|---:|---:|---:|---:|---|
| Month 3 flat MLP | Measured | Measured | 1 | Measured | Measured | Baseline |
| Wider flat MLP | Measured | Measured | 1 | Measured | Measured | All context mixed at once |
| Hierarchical model | Measured | Measured | Measured | Measured | Measured | Pairwise growth of receptive field |

Explain performance differences without claiming causation from one run alone.

### Intentional break

Swap the batch and time dimensions before BatchNorm or normalize across the wrong axes. Record which statistics appear plausible and which semantic meaning is wrong. This exercise is about preventing “shape-valid but conceptually invalid” code.

### Deliverable

- `makemore_deep/wavenet.py`.
- Layer tests.
- Comparison table and generated names.

### Exit criteria

- The hierarchical model trains and generates.
- Every shape and receptive field is explainable.
- You can identify the limitation: the pattern of communication is fixed by the hierarchy rather than selected dynamically from content.
