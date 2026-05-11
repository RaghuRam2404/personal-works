# Week 10 — "Attention Is All You Need" — The Paper, Line by Line

## Learning Objectives

By end of this week, you will be able to:

- Derive scaled dot-product attention on paper and explain the `1/sqrt(d_k)` scaling term
- Describe multi-head attention: why multiple heads, what each head sees, how outputs are concatenated
- Explain sinusoidal positional encoding and why it allows generalization to unseen lengths
- Implement the complete encoder-decoder Transformer from The Annotated Transformer
- Explain the role of each sublayer (self-attention, cross-attention, FFN) in the encoder and decoder
- Identify the key differences between the 2017 Transformer and RNN-based seq2seq

## Time Allocation (6–8 hrs)

| Activity | Time |
|---|---|
| Read "Attention Is All You Need" (first pass, cover-to-cover) | 1.5 hrs |
| Read The Annotated Transformer (nlp.seas.harvard.edu) | 1 hr |
| Watch Yannic Kilcher full video (28 min) + 3B1B videos (~53 min total) | 1.5 hrs |
| Type out Annotated Transformer implementation | 2.5 hrs |
| Train on toy task, write notes, commit | 1 hr |

---

## Concepts

### Why "Attention Is All You Need"

The 2017 paper by Vaswani et al. proposed dropping RNNs entirely. Recurrence requires sequential processing — you cannot compute hidden state `h_t` until `h_{t-1}` is ready, which limits parallelism during training. The Transformer replaces recurrence with self-attention: every position in the sequence can attend to every other position in a single matrix operation. This unlocks massive parallelism and enables training on far larger datasets.

### Scaled Dot-Product Attention

Given queries Q, keys K, values V (all matrices):

```
Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V
```

Each row of Q is a "question" — what information am I looking for? Each row of K is a "label" — what information do I contain? Each row of V is the actual content — what do I return if selected? The dot product `Q K^T` computes compatibility scores for all (query, key) pairs simultaneously. Softmax normalizes these into a probability distribution. The result is a weighted sum of the values.

**Why divide by `sqrt(d_k)`?** When `d_k` is large (e.g., 64), the dot products grow large in magnitude. This pushes the softmax into regions where its gradient is near zero (saturation). Dividing by `sqrt(d_k)` keeps the pre-softmax values in a range where gradients flow well. If you don't scale, training stalls because softmax gradients vanish. You can verify this: for random unit vectors in `d_k` dimensions, the expected value of the dot product is 0 and the variance is `d_k`. Dividing by `sqrt(d_k)` normalizes variance to 1.

### Multi-Head Attention

Instead of computing one attention function, the Transformer linearly projects Q, K, V into `h` different lower-dimensional spaces, computes attention in each, and concatenates:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W_O
head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
```

Why multiple heads? Each head can specialize in a different type of relationship. In practice, heads learn diverse behaviors: some attend to nearby tokens (local), some attend to syntactic dependencies (e.g., verb-object), some attend to coreference. If you use only one head, the model must compromise. Multiple heads with the same total compute allow specialization.

In the paper: `d_model = 512`, `h = 8`, so each head has `d_k = d_v = 64`. The total parameter count is the same as a single head with `d_model` dimensions — the projection matrices absorb the cost.

### The Encoder

The encoder has `N = 6` identical layers. Each layer has two sublayers:
1. Multi-head self-attention (every position attends to every other position)
2. Position-wise feed-forward network (FFN): `FFN(x) = max(0, x W_1 + b_1) W_2 + b_2`

Each sublayer uses a residual connection and layer normalization:
```
output = LayerNorm(x + Sublayer(x))
```

The FFN expands dimension by 4x (512 → 2048 → 512). The intuition: the FFN acts as a key-value memory store — it adds information that cannot be expressed by attention alone.

### The Decoder

The decoder also has 6 layers, but each layer has three sublayers:
1. Masked multi-head self-attention (causal — position `i` can only attend to positions 1..i)
2. Cross-attention (queries from decoder, keys and values from encoder output)
3. FFN

The causal mask is applied by setting future positions to `-inf` before softmax. This ensures autoregressive generation — the model predicts position `i` without seeing positions `i+1, i+2, ...`.

Cross-attention is what connects encoder and decoder: the decoder queries "what encoder information do I need at this step?" This is directly analogous to Bahdanau's context vector from Week 9 — but now it's implemented with scaled dot-product attention and fully parallelizable.

### Positional Encoding

Self-attention is permutation-invariant — if you shuffle the input tokens, the output is the same tokens shuffled in the same way. The model has no built-in notion of position. Positional encodings add position information:

```
PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
```

These are fixed (not learned in the original paper). The sinusoidal form has a key property: the positional encoding for position `pos + k` is a linear function of the encoding for position `pos`, which means the model can learn to attend by relative offset. It also generalizes to sequence lengths longer than those seen during training.

### From Week 9 to Week 10

In Week 9, attention was "additive" and applied only from decoder to encoder (cross-attention). This week introduces self-attention (every layer attends to itself) and replaces the entire RNN with attention. The Bahdanau alignment matrix was computed once per translation. In a Transformer, every layer of every block computes an attention matrix — the model learns hierarchical representations through stacked attention.

### SQL Connection

In text-to-SQL, encoder self-attention allows schema tokens to interact with question tokens in early layers. Cross-attention aligns the decoder's SQL generation with relevant question/schema tokens. BRIDGE (a strong text-to-SQL system) is a Transformer encoder with schema linking built on exactly these mechanisms. You will revisit this in Phase 5.

## Connections

This week is the architectural keystone of the entire course. It builds directly on Week 9 (Bahdanau attention — you now see why self-attention generalizes that idea: instead of attending across two sequences, every token attends to every other token in one sequence). It also builds on every PyTorch fundamental from Phase 1 — `nn.Linear`, `nn.LayerNorm`, broadcasting, and the autograd machinery you internalized in Weeks 1–5 are all in the encoder-decoder you implement this week.

What depends on this material: literally every remaining week of the course. Week 11 strips the encoder to give you GPT-style decoder-only transformers. Week 12 swaps post-LN for pre-LN, LayerNorm for RMSNorm, learned positional embeddings for RoPE, and ReLU FFN for SwiGLU — those swaps only make sense once you can articulate exactly what they replace from the original 2017 paper. Week 14's `modeling_llama.py` annotation will feel like a code review of variations on what you wrote this week. And every fine-tuning, quantization, and RL technique in Phases 4–5 operates on the architecture you implement here. If any concept in this week is shaky, fix it now — the cost of weakness compounds for 60+ weeks.

## Common Misconceptions / Pitfalls

- **Thinking multi-head attention is more expensive.** It's not — the projections reduce dimension per head so total compute is equivalent.
- **Forgetting the causal mask in the decoder.** If you don't mask future positions in decoder self-attention, the model cheats during training and generates nonsense at inference time.
- **Not applying residual connections.** Without `x + Sublayer(x)`, deep transformers fail to train — the residual is what makes gradients flow through 6+ layers.
- **Confusing d_model, d_k, d_v.** In the original paper: d_model=512, d_k=d_v=64, h=8. The projection matrices `W^Q, W^K, W^V` have shape `[d_model, d_k]`. Don't conflate them.
- **Layer norm position.** The original paper uses post-LN (after sublayer). Modern transformers use pre-LN (before sublayer). The Annotated Transformer follows the original; be aware the convention changed.
