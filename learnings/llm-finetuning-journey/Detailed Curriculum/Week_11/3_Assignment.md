# Week 11 Assignment — Building nanoGPT and Training on SQL

## Setup Checklist

- [ ] Colab Free (this fits comfortably for the Shakespeare run; the SQL run may need a GPU)
- [ ] W&B project `week-11-nanogpt`
- [ ] GitHub branch `week-11-nanogpt`
- [ ] Watch the full Karpathy video (1h56m) BEFORE coding — code along during the video

---

## Task 1 — Implement nanoGPT from Scratch (Code Along)

**Goal:** Type out Karpathy's decoder-only GPT implementation by following the video. Every line must be typed by you — no copy-paste from the nanoGPT repo.

**Requirements:**
- Implement `CausalSelfAttention(nn.Module)`:
  - Takes `(x)` of shape `[B, T, C]`
  - Projects to Q, K, V
  - Computes scaled dot-product attention with a causal mask (use `torch.tril`)
  - Applies dropout on attention weights
  - Returns `[B, T, C]`
- Implement `MLP(nn.Module)`:
  - Linear(C, 4C) → GELU → Linear(4C, C) → Dropout
- Implement `Block(nn.Module)`:
  - Pre-LN: `x = x + attn(ln1(x))`
  - Pre-LN: `x = x + mlp(ln2(x))`
- Implement `GPT(nn.Module)`:
  - Token embedding `wte`: `[vocab_size, n_embd]`
  - Position embedding `wpe`: `[block_size, n_embd]`
  - N identical `Block` layers
  - Final `LayerNorm`
  - LM head: `Linear(n_embd, vocab_size, bias=False)` with weights tied to `wte`
  - `forward(idx, targets=None)` returns `(logits, loss)`
  - `generate(idx, max_new_tokens, ...)` for autoregressive generation

**Hyperparameters for Shakespeare run:**
- `n_layer=6, n_head=6, n_embd=384, block_size=256, dropout=0.2`
- Batch size: 64, learning rate: 3e-4, max_iters: 5000
- Optimizer: AdamW

**Requirements (deliverable checks):**
- `model.generate(...)` produces readable (though imperfect) Shakespeare-style text
- Val loss < 1.65 after 5000 iters on Tiny Shakespeare (character-level)
- W&B run `week-11-shakespeare` logged

**Deliverable:** `model.py` in GitHub commit `week-11-nanogpt`

---

## Task 2 — Train on SQL Corpus

**Goal:** Retrain (from scratch, new weights) your nanoGPT on a SQL corpus. Compare outputs.

**Requirements:**
- Obtain a SQL corpus. Two options:
  - Option A (recommended): Download Spider dataset's training SQL queries (only the SQL field, not the natural language). Available at [Yale Spider dataset](https://yale-seas.github.io/spider/). Extract ~10,000 SQL queries.
  - Option B: Generate a synthetic SQL corpus using Python — randomly combine SELECT / FROM / WHERE / JOIN / GROUP BY patterns with fake table/column names. Aim for 500KB+ of text.
- Preprocess: concatenate all SQL queries separated by `\n`, character-tokenize
- Use smaller model for SQL (it's a smaller corpus): `n_layer=4, n_head=4, n_embd=256, block_size=128`
- Train for at least 3000 iterations
- Log to W&B project `week-11-nanogpt` (run name: `sql-run`)
- Generate 5 sample SQL snippets from the trained model
- Commit generated samples as `sql_samples.txt`

**Acceptance criteria:**
- Generated SQL contains recognizable SQL keywords (SELECT, FROM, WHERE, JOIN) in syntactically plausible positions
- Val loss < 1.5 on SQL corpus

**Deliverable:** `sql_samples.txt` + W&B run in GitHub commit.

---

## Task 3 — Answer These in Code Comments

Add a `# NOTES` section at the top of your `model.py` answering:

1. Why does `nn.Embedding` and the LM head share weights? What happens if you don't tie them?
2. Why does the position embedding `wpe` have dimension `[block_size, n_embd]` rather than `[T, n_embd]`?
3. Why do we use GELU instead of ReLU in the MLP? (Look up the GELU paper or Karpathy's explanation.)

**Deliverable:** Comments in `model.py` — 2–4 sentences per question.

---

## Stretch Goals

- Reduce `n_embd` to 64 and train on the SQL corpus. At what model size does generated SQL become unrecognizable?
- Implement top-k sampling in your `generate()` method. Compare outputs with k=1 (greedy), k=10, k=50.
- Add a training curve visualization: plot train loss vs. val loss in matplotlib. Is the model overfitting or underfitting?
