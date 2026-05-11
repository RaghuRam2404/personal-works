# Week 11 — Decoder-Only Transformers and the GPT Family

## Learning Objectives

By end of this week, you will be able to:

- Explain why GPT dropped the encoder and what was gained by doing so
- Describe causal language modeling as a pretraining objective and why it requires no labeled data
- Implement a decoder-only transformer (nanoGPT style) from scratch in PyTorch
- Explain weight tying between the embedding table and the LM head
- Describe the residual stream view of transformer computation
- Train nanoGPT on Tiny Shakespeare and on a SQL corpus; compare generated samples

## Time Allocation (6–8 hrs)

| Activity | Time |
|---|---|
| Read GPT-1 and GPT-2 papers (focus on architecture sections) | 1 hr |
| Skim GPT-3 architecture + scaling section | 0.5 hrs |
| Watch Karpathy nanoGPT video (1h56m), code along | 4 hrs |
| Retrain on SQL corpus, generate samples, commit | 1.5 hrs |

---

## Concepts

### Why Drop the Encoder?

The original Transformer (Week 10) was designed for translation — a task with a clear source and target sequence. But language modeling — predicting the next token given a context — doesn't need an encoder. There's no separate "source" to compress. The input is the context itself, and the model autoregressively predicts what comes next.

GPT-1 (2018) made this architectural choice deliberately. By training a decoder-only model on a massive text corpus with a simple next-token prediction objective, the model learns rich representations that transfer to downstream tasks with minimal task-specific fine-tuning. The encoder is not dropped out of laziness; dropping it forces the model to develop a unified representational space where understanding and generation are the same operation.

### The Architecture Difference

In the encoder-decoder Transformer:
- Encoder: bidirectional (every token sees every other token)
- Decoder: causal (each token sees only past tokens) + cross-attention to encoder

In a decoder-only Transformer:
- Single stack of causal self-attention layers
- No cross-attention (there is no encoder)
- Every layer is identical: causal MHA → LayerNorm → FFN → LayerNorm (with residuals)

The causal mask from Week 10's decoder is the key mechanism. Every token can only attend to tokens at or before its own position. This enforces the autoregressive property during both training and inference.

### Causal Language Modeling Objective

Given a sequence of tokens `x_1, x_2, ..., x_T`, the model is trained to minimize the negative log-likelihood of the next token at each position:

```
L = - (1/T) * sum_t log P(x_t | x_1, ..., x_{t-1})
```

This is also called cross-entropy loss computed over shifted targets. In code:

```python
logits, _ = model(x)          # [B, T, vocab_size]
loss = F.cross_entropy(
    logits[:, :-1, :].reshape(-1, vocab_size),
    x[:, 1:].reshape(-1)
)
```

You shift the targets by 1: the model sees token 1 and predicts token 2, sees tokens 1-2 and predicts token 3, etc. All T-1 predictions are computed in a single forward pass due to the causal mask. This is extremely data-efficient — a single sequence of length T provides T-1 training examples.

### The Residual Stream View

A powerful way to think about transformer computation (popularized by Anthropic's interpretability work) is the residual stream. The input to each layer is a vector `x`. The layer computes some "delta" and adds it to `x`:

```
x = x + MHA(LN(x))      # attention adds to the stream
x = x + FFN(LN(x))      # FFN adds to the stream
```

The residual stream `x` persists throughout the network. Each sublayer reads from the stream, computes a contribution, and writes it back. The final `x` is the superposition of everything every layer wrote. This view helps explain why residual connections are essential (without them, there is no stream to write to) and why early layers often handle low-level syntactic features while late layers handle high-level semantics — they're writing progressively abstract information to the same stream.

### Weight Tying

In decoder-only transformers, the embedding matrix `E` (shape `[vocab_size, d_model]`) and the LM head (a linear layer that projects `d_model` → `vocab_size` to produce logits) share the same weights. This means `W_LM_head = E`.

Why? Two reasons:
1. **Parameter efficiency:** The embedding matrix is large (for GPT-2: 50,257 × 768 ≈ 38M parameters). Sharing saves 38M parameters.
2. **Semantic coherence:** Tokens that are semantically similar should have similar embeddings AND similar logit profiles. Sharing the same matrix enforces this consistency — the model learns to embed tokens in a space where cosine similarity in embedding space corresponds to substitutability in generation.

In nanoGPT:
```python
self.transformer.wte = nn.Embedding(vocab_size, n_embd)
self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
# tie weights:
self.lm_head.weight = self.transformer.wte.weight
```

### GPT-1 → GPT-2 → GPT-3 Progression

| Model | Year | Params | Tokens | Key change |
|---|---|---|---|---|
| GPT-1 | 2018 | 117M | ~5B | First decoder-only pretraining |
| GPT-2 | 2019 | 1.5B | ~40B | Pre-LN, larger scale, no supervised fine-tuning |
| GPT-3 | 2020 | 175B | 300B | Few-shot prompting emerges at scale |

GPT-2 made one important architectural change from GPT-1: it moved LayerNorm to the input of each sublayer (Pre-LN) and added a final LayerNorm before the LM head. Pre-LN makes training more stable (see Week 12 for details). Everything else in the architecture is the same — just bigger.

### Training nanoGPT on SQL

Karpathy's nanoGPT trains a character-level model on Tiny Shakespeare. This week you will also train it on a SQL corpus (you can use the SQL queries from the Spider dataset's training set, or generate a simple corpus of SQL queries). Training on SQL is not fine-tuning — you are pretraining a character-level model from scratch. The model will learn SQL syntax, keyword usage, and table/column name patterns from raw character prediction.

The SQL character set is smaller than Shakespeare (~60–80 unique characters vs. ~65 for Shakespeare). Training should converge faster. Generated SQL will not be semantically correct (the model has no database schema to reference), but it should be syntactically plausible — recognizable SELECT/FROM/WHERE structure.

## Connections

This week builds directly on Week 9 (Bahdanau attention and the encoder-decoder pattern) and Week 10 (the full Transformer architecture). Week 9 established why attention works; Week 10 showed how encoder and decoder stacks compose. Here you strip the encoder away entirely and understand why that is a feature, not a deficiency. The causal mask you implemented in Week 10's decoder is the same mechanism driving the whole model this week.

Week 12 depends on this foundation: you will add modern architectural improvements — RoPE, GQA, RMSNorm, and SwiGLU — to the decoder-only skeleton you build here. Week 14 builds further, mapping the production LLaMA code to the same architecture you understand today. Week 15 uses everything you learn in Weeks 11–14 to reproduce GPT-2 124M from scratch — that run will only make sense if you have internalized the weight-tying, causal masking, and residual stream concepts from this week.

## Common Misconceptions / Pitfalls

- **"Decoder-only" means missing half the architecture.** The decoder-only model is a complete, self-sufficient architecture. The encoder is not "missing" — it was never needed for the pretraining objective.
- **Confusing Pre-LN and Post-LN in nanoGPT.** nanoGPT uses Pre-LN. If you're following along with older GPT-1 code, verify which convention is used.
- **Not tying weights.** If you forget weight tying, the model uses ~38M extra parameters and tends to be less coherent. The symptom is subtle — training still works, but generation quality is slightly lower.
- **Training on full sequences without proper padding.** For a character-level model, if you concatenate all text into one long stream and chunk it, there's no padding needed. If you use variable-length examples, you need careful masking.
- **Forgetting to set model to eval mode during inference.** Dropout is active in train mode. Always `model.eval()` and `torch.no_grad()` before generating text.
