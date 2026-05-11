# Week 10 Assignment — Implementing the Annotated Transformer

## Setup Checklist

- [ ] PyTorch ≥ 2.0
- [ ] `torchtext` or manual data loading (the Annotated Transformer uses raw Python)
- [ ] Colab Free tier (encoder-decoder Transformer on a toy task fits in free RAM)
- [ ] GitHub branch `week-10-annotated-transformer`
- [ ] W&B project `week-10-transformer`

---

## Task 1 — Implement Scaled Dot-Product Attention

**Goal:** A standalone function (not a module) that implements the core attention computation.

**Requirements:**
- Signature: `scaled_dot_product_attention(Q, K, V, mask=None) -> (output, attn_weights)`
- Q: `[batch, heads, seq_len_q, d_k]`, K: `[batch, heads, seq_len_k, d_k]`, V: `[batch, heads, seq_len_k, d_v]`
- Scale by `1 / sqrt(d_k)` before softmax
- If `mask` is provided (bool, True = ignore), apply `masked_fill(-1e9)` before softmax
- Unit test: verify output shape is `[batch, heads, seq_len_q, d_v]` and `attn_weights.sum(dim=-1)` equals 1 everywhere
- Write a separate test: pass in Q=K=V of shape `[1, 1, 5, 64]` with no mask. Verify the attention matrix is symmetric.

**Deliverable:** `attention_fn.py` with unit tests passing.

---

## Task 2 — Implement Multi-Head Attention

**Goal:** `MultiHeadAttention(nn.Module)` that wraps scaled dot-product attention with projection layers.

**Requirements:**
- Constructor: `MultiHeadAttention(d_model, num_heads)`
- Four linear layers: `W_Q`, `W_K`, `W_V` (each `[d_model, d_k]`), `W_O` (`[h * d_v, d_model]`)
- Assert `d_model % num_heads == 0` and derive `d_k = d_v = d_model // num_heads`
- Forward: project → split heads → attend → concat → project out
- Must pass through a causal mask correctly (test with a sequence of length 5; check that future positions get zero attention weight)

**Deliverable:** `multi_head_attention.py`

---

## Task 3 — Assemble and Train the Full Encoder-Decoder Transformer

**Goal:** Type out the complete Annotated Transformer implementation from [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/). Train on a toy copy task (copy the input sequence) to verify the architecture works.

**Requirements:**
- Implement (type, do not copy-paste): `PositionalEncoding`, `EncoderLayer`, `Encoder`, `DecoderLayer`, `Decoder`, `Generator`, and the full `EncoderDecoder` model
- Use `d_model=128`, `num_heads=4`, `num_layers=2`, `d_ff=256`, `dropout=0.1` (reduced from the paper for speed)
- Train on the copy task: input is a random sequence of integers [1..10], target is the same sequence
- Train for 2000 steps, batch size 32
- Log loss to W&B every 100 steps
- Acceptance criteria: model achieves near-perfect accuracy (>99%) on the copy task within 2000 steps

**Hints:**
- The copy task sounds trivial but it's a real debugging tool — if your masked decoder self-attention is broken, the model will not learn to copy faithfully.
- Label smoothing is used in the paper (epsilon=0.1). It is worth implementing — it prevents overconfidence and improves generalization.
- Use the Adam optimizer with the warmup learning rate schedule from the paper: `lr = d_model^{-0.5} * min(step^{-0.5}, step * warmup_steps^{-1.5})` with `warmup_steps=400`.

**Deliverable:** GitHub commit `week-10-annotated-transformer` with `transformer.py` and a W&B run showing loss to near-zero.

---

## Task 4 — Derive `1/sqrt(d_k)` Scaling on Paper (Non-Coding)

**Goal:** Convince yourself (and document) why the scaling factor is necessary.

**Requirements:**
- Write a short experiment in a Jupyter notebook or script: sample random Q and K matrices with entries from N(0,1), compute Q K^T / sqrt(d_k), measure the std of the pre-softmax scores for d_k = 4, 16, 64, 256
- Show (numerically) that without scaling, the std of scores grows as sqrt(d_k)
- Show that with scaling, the std stays close to 1 regardless of d_k
- Write 3–4 sentences explaining why this matters for softmax (include the word "saturation")

**Deliverable:** `scaling_experiment.py` (or `.ipynb`) with printed results. Commit to repo.

---

## Stretch Goals

- Replace the toy copy task with a toy English→French vocabulary (50 word pairs). Can the model learn to translate?
- Visualize the attention weights of all 4 heads in layer 2 on a single input. Do different heads attend to different positions?
- Train on a tiny SQL dataset (Spider train split, first 1000 examples, encode as plain text sequences). Observe if the model can produce any SQL-like outputs.
