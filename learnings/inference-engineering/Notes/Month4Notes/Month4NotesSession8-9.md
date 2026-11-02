[← Month 4 notes](<../Month4Notes.md>) · [← Session 5-7](<Month4NotesSession5-7.md>) · [Session 10 →](<Month4NotesSession10.md>)

## Session 8,9

Answers are based on the active model in [5 wavenet.py:223-229](../../Month%204/5%20wavenet.py#L223-L229).

**1. How many aggregation levels are needed to combine 8 positions pairwise?**
It takes 3 levels, because log₂(8) = 3. Each level halves the number of positions: 8 → 4 → 2 → 1. The model has three `FlattenConsecutive(2)` + `Linear` + `BatchNorm1D` + `Tanh` blocks, one per level.

**2. What is the receptive field after each level?**
- Level 1: 2 characters. Each output position sees 2 adjacent embeddings.
- Level 2: 4 characters. Each position combines two level-1 outputs.
- Level 3: 8 characters. This is the whole `block_size`, so it is the full context.

The receptive field doubles at each level.

**3. Why can the same transformation be reused across positions at one level?**
`FlattenConsecutive(2)` reshapes `(B, T, C)` into `(B, T/2, 2C)`. Each position then holds one pair, laid out the same way. `Linear` computes `x @ w` on the last axis only, so one weight matrix `(2C, n_hidden)` is applied to every pair. The positions share weights, as a conv with kernel size 2 and stride 2 would. This works for two reasons:
- Every pair has the same format, with the left and right element in the same slots.
- The rule for combining two neighbours does not depend on where in the word they sit.

`BatchNorm1D` also supports this. For 3D input it normalizes over `dim=(0,1)`, so batch and time share one set of statistics per channel.

**4. Why is hierarchical aggregation not yet attention?**
- **The grouping is fixed.** Only adjacent pairs `(0,1)`, `(2,3)`, … are combined. Position 0 can't look at position 5 until the last level, and only through the fixed tree.
- **The weights are content-independent.** `w` is the same whatever the characters are. Attention computes weights from the tokens themselves, through query·key similarity and a softmax.
- **There is no selective routing.** The model can't choose which earlier character matters for the current one. It can only mix them with learned, fixed linear maps.
- **The combination is concatenate then linear.** Attention takes a data-dependent weighted sum of value vectors.

So this is a fixed tree of shared-weight local mixers. It is like a dilated conv, not a content-addressed lookup.

**5. Which tensor axis represents time/space at every level?**
It is axis 1, the `T` in `(B, T, C)`. Axis 0 is the batch and the last axis is channels or features. Tracing the shapes:

| Step | Shape |
|---|---|
| Embedding | `(B, 8, 10)` |
| Flatten(2) → Linear | `(B, 4, 20)` → `(B, 4, 100)` |
| Flatten(2) → Linear | `(B, 2, 200)` → `(B, 2, 100)` |
| Flatten(2) → Linear | `(B, 1, 200)` → squeezed to `(B, 200)` → `(B, 100)` |
| Final Linear | `(B, 27)` |

The time axis shrinks 8 → 4 → 2 → 1 as it is folded into the channel axis. At level 3, `FlattenConsecutive` squeezes it away (lines 158-159), so the final `Linear` sees a 2D tensor.
