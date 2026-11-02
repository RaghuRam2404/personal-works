[← Month 4 overview](<../Month4.md>) · [← Session 13](<Month4CurrSession13.md>) · [Session 15 →](<Month4CurrSession15.md>)

## Session 14 — Build GPT lecture

### Objective

Understand how embeddings, causal self-attention, feed-forward networks, residual streams, LayerNorm, and repeated blocks form a decoder-only Transformer.

### Material

Watch **Let's build GPT: from scratch, in code, spelled out**.

- Video: <https://www.youtube.com/watch?v=kCc8FmEb1nY>
- Reference repository: <https://github.com/karpathy/ng-video-lecture>
- `bigram.py`: <https://github.com/karpathy/ng-video-lecture/blob/master/bigram.py>
- `gpt.py`: <https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py>

The reference repository contains the code produced during the lecture and its Git history, which can be inspected after your implementation attempt.[cite:188] The lecture's published model uses token and positional embeddings, causal multi-head attention, feed-forward layers, residual connections, LayerNorm, and repeated Transformer blocks.[cite:189]

### Watching checkpoints

Pause and answer before each addition:

1. What limitation of the bigram baseline is being fixed?
2. Why does averaging previous token representations not provide content-dependent retrieval?
3. Why are positional embeddings needed when attention scores depend only on token representations?
4. Why does each position need both communication and private computation?
5. Why do attention and feed-forward sublayers preserve `[B, T, C]`?
6. Why are residual connections possible only when input/output shapes match?
7. Why is LayerNorm preferable to BatchNorm for this per-token Transformer path?
8. Why is a final vocabulary projection required?

LayerNorm calculates statistics over specified final feature dimensions and uses input statistics in both training and evaluation, unlike the running-statistic behavior of BatchNorm.[cite:206]

### Transformer block diagram

Draw and annotate the pre-normalization block used in the lecture:

```text
x ────────────────────────────────┐
│                                 │
└→ LayerNorm → causal MHA → + ────┘ = x'

x' ───────────────────────────────┐
│                                 │
└→ LayerNorm → FeedForward → + ───┘ = output
```

For every branch, identify:

- Shape.
- Learned parameters.
- Whether tokens communicate across positions.
- Whether the operation acts independently per position.

### Architecture ledger

For input IDs `[B, T]`, derive:

```text
token embeddings:        [B, T, C]
position embeddings:     [T, C] → broadcast to [B, T, C]
residual stream:         [B, T, C]
per-head q/k/v:          [B, H, T, D]
attention scores:        [B, H, T, T]
attention output:        [B, T, C]
feed-forward hidden:     [B, T, 4C]
feed-forward output:     [B, T, C]
final normalized state:  [B, T, C]
logits:                  [B, T, V]
loss input:              [B*T, V]
targets:                 [B*T]
```

### Assignment M4.14

Write one sentence for each component answering “what unique job does this do?”

- Token embedding.
- Position embedding.
- Query projection.
- Key projection.
- Value projection.
- Causal mask.
- Multiple heads.
- Output projection.
- LayerNorm.
- Residual connection.
- Feed-forward network.
- Dropout.
- Vocabulary projection.

### Deliverable

- Lecture notes.
- Annotated block diagram.
- Full architecture shape ledger.
- Component-purpose table.

### Exit criteria

- You can explain a decoder block as communication plus per-token computation.
- You can derive all major tensor shapes without executing code.
- You can explain why BatchNorm from earlier sessions is not simply inserted into the GPT block.
