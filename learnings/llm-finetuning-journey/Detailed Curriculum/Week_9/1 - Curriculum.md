# Week 9 — The Original Attention Mechanism (Bahdanau, 2014)

## Learning Objectives

By end of this week, you will be able to:

- Explain why attention was invented and what problem in seq2seq models it solved
- Derive the Bahdanau additive attention score function on paper
- Implement a seq2seq LSTM with Bahdanau attention in PyTorch from scratch
- Visualize attention weight matrices and interpret what the model "looks at"
- Distinguish additive (Bahdanau) from multiplicative (Luong) attention
- Describe the alignment mechanism and why it enables translation of long sentences

## Time Allocation (6–8 hrs)

| Activity | Time |
|---|---|
| Read Bahdanau paper (1409.0473) with notes | 2.5 hrs |
| Read Lilian Weng blog post | 0.5 hrs |
| Watch Yannic Kilcher intro (first 10 min) | 0.25 hrs |
| Implement seq2seq + attention in PyTorch | 2.5 hrs |
| Plot attention heatmap, write commit + notes | 0.75 hrs |

---

## Concepts

### The Problem with Vanilla Seq2Seq

In 2014, the dominant architecture for machine translation was sequence-to-sequence (seq2seq) with an encoder RNN and a decoder RNN. The encoder processes the input sequence one token at a time and compresses the entire source sentence into a single fixed-size vector — the context vector. The decoder then uses that vector to generate the target sequence.

This works for short sentences but degrades badly as sentence length grows. The information bottleneck is severe: a single vector of dimension 256 or 512 must somehow encode "The bank near the river flooded yesterday, causing significant damage to the surrounding farmland" before the decoder generates a word. RNNs also suffer from vanishing gradients over long sequences, so even if the vector were large, the encoder's hidden state for position 1 is nearly washed out by position 30.

Bahdanau et al. (2014) proposed a simple fix: instead of forcing the decoder to rely on one fixed context vector, let the decoder look back at all encoder hidden states and dynamically compute a weighted sum. The weights are learned and they change at every decoder step — so when generating the French word for "bank", the decoder can attend strongly to the English word "bank" rather than to filler tokens. This is attention.

### The Bahdanau Architecture

The full architecture has three parts:

**Encoder:** A bidirectional LSTM (or GRU) processes the input sequence `x_1, ..., x_T`. For each position `j`, you get a forward hidden state and a backward hidden state. Bahdanau concatenates them into an annotation vector `h_j`. Using a bidirectional encoder means `h_j` captures context from both directions around position `j`, not just what came before.

**Alignment model (the attention scorer):** At each decoder time step `i`, you compute an alignment score `e_{i,j}` for every encoder position `j`:

```
e_{i,j} = v_a^T * tanh(W_a * s_{i-1} + U_a * h_j)
```

where `s_{i-1}` is the decoder hidden state from the previous step, and `W_a`, `U_a`, `v_a` are learned weight matrices. This is called additive attention because the decoder state and encoder annotation are summed (after linear projection) inside the tanh. The entire function is a small single-hidden-layer MLP.

**Context vector:** Normalize the scores with softmax to get attention weights `alpha_{i,j}`:

```
alpha_{i,j} = softmax(e_{i,j})   over all j
```

Then compute the context vector `c_i` as the weighted sum of encoder annotations:

```
c_i = sum_j ( alpha_{i,j} * h_j )
```

**Decoder:** A standard LSTM that receives `c_i` concatenated with the embedding of the previous output token at each step.

The key insight: `alpha_{i,j}` is a soft alignment. It tells you, "when generating target word `i`, how much attention should I pay to source word `j`?" If you print this matrix, you get a heatmap that looks like a (noisy) diagonal for language pairs with similar word order — and an off-diagonal pattern for pairs like English-French where adjective/noun order swaps.

### Additive vs. Multiplicative Attention

Bahdanau's score function requires two weight matrices and a vector, plus a tanh. Luong et al. (2015) proposed a simpler multiplicative (dot-product) form:

```
e_{i,j} = s_i^T * h_j
```

or with a learned matrix:

```
e_{i,j} = s_i^T * W * h_j
```

Multiplicative attention is cheaper computationally and became dominant. The Transformer (Week 10) uses scaled dot-product attention — a refinement of multiplicative attention. Bahdanau's additive form is slower but historically important: it was the mechanism that broke the fixed context-vector bottleneck.

### Why This Matters for SQL Generation

When generating SQL from a natural language question, attention plays a critical role: the model must align the word "city" in the question to the column `cities.name` in the schema. Modern text-to-SQL systems like BRIDGE and RAT-SQL are built on this core idea — the model learns soft alignments between question tokens and schema elements. You are building the intuition that will directly underpin your Phase 6 fine-tuning work.

### String-Reversal as a Sanity Check

For your assignment, you will train a seq2seq model to reverse short strings (e.g., "hello" → "olleh"). This is a clean task because the expected attention pattern is perfectly anti-diagonal — the model should attend to the last input character when generating the first output character. If your attention heatmap shows this anti-diagonal pattern, your implementation is correct. If it doesn't, something is wrong with your attention computation or your masking.

## Connections

**Building on:** Week 8 (LSTM training, backprop through time, gradient flow) and linear algebra intuition for matrix operations.

**Used in:** Week 10 (attention directly generalizes to self-attention in transformers), Week 13 (KV cache is directly derived from the "keys and values" abstraction you build here), Week 14 (LLaMA's attention is a direct descendant of this mechanism).

## Common Misconceptions / Pitfalls

- **Confusing context vector and hidden state.** The context vector `c_i` is newly computed at each decoder step. The decoder hidden state `s_i` is the running state. They are different tensors concatenated as input to the decoder.
- **Not masking padding tokens.** If your source sentence is padded to length 30 but the actual sentence is length 10, you must set `e_{i,j} = -inf` for `j > 10` before softmax. Otherwise the model attends to padding garbage.
- **Bidirectional encoder in inference.** The bidirectional encoder only runs during training/encoding. The decoder is unidirectional (causal). Students sometimes try to make the decoder bidirectional, which is a category error.
- **Forgetting to detach hidden states across batches.** When you process multiple batches, don't carry hidden states across batch boundaries or you'll backprop into the previous batch.
- **Attention weights don't sum to 1 per row.** They must — the softmax is taken across all source positions `j` for a fixed target position `i`. If you apply softmax along the wrong dimension, training will appear to work but attention maps will be incoherent.
