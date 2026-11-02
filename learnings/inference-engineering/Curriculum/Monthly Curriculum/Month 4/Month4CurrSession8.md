[← Month 4 overview](<../Month4.md>) · [← Session 7](<Month4CurrSession7.md>) · [Session 9 →](<Month4CurrSession9.md>)

## Session 8 — WaveNet-style hierarchy

### Objective

Understand why a flat context MLP scales poorly and how hierarchical pairwise aggregation expands the receptive field gradually.

### Material

Watch **Building makemore Part 5: Building a WaveNet**.

- Video: <https://youtu.be/t3YJ5hKiMQ0>
- Official notebook: <https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part5_cnn1.ipynb>

The official course describes this model as a deeper tree-like architecture similar to WaveNet, while emphasizing `torch.nn`-style layers, multidimensional shapes, and normal development workflow.[cite:131]

### Pre-watch reasoning

For context length 8, compare:

- Flat MLP: concatenate all 8 embeddings and mix them in one linear layer.
- Hierarchical model: combine adjacent pairs, then pairs of pair-representations, then the final pair.

Answer:

1. How many aggregation levels are needed to combine 8 positions pairwise?
2. What is the receptive field after each level?
3. Why can the same transformation be reused across positions at one level?
4. Why is hierarchical aggregation not yet attention?
5. Which tensor axis represents time/space at every level?

### Required concepts

- Receptive field.
- Hierarchical aggregation.
- Weight sharing over positions.
- `FlattenConsecutive`.
- Rank-3 activations `[B, T, C]`.
- BatchNorm over multi-dimensional activations.
- Shape-preserving linear transformation over the last dimension.
- Relationship between tree depth and context length.

### Assignment M4.8

Draw the path for context positions 0–7 through three pairwise aggregation levels. For the final representation, identify exactly which input positions can influence it.

Write the complete symbolic shape sequence for:

```text
[B, 8]
→ embedding
→ first pair flatten
→ linear
→ BatchNorm
→ tanh
→ second pair flatten
→ linear
→ BatchNorm
→ tanh
→ third pair flatten
→ linear
→ vocabulary logits
```

### Deliverable

- Lecture notes.
- Receptive-field diagram.
- Symbolic shape ledger.

### Exit criteria

- You can explain the difference between flat, hierarchical, recurrent, and attention-based context mixing at a high level.
- Every hierarchy level's receptive field and tensor shape can be derived.
