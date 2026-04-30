# Week 20 Assignment — Pretraining Setup

## Setup Checklist

- [ ] `pip install tokenizers datasets torch accelerate wandb numpy`
- [ ] W&B account created and `wandb login` completed
- [ ] Colab Pro session with A100 runtime (needed for Week 21; verify access now)
- [ ] GitHub repo with `pretrain-50m/` directory
- [ ] HuggingFace account for dataset access

---

## Task 1 — Train a BPE Tokenizer

**Goal:** Train a 32K-vocabulary BPE tokenizer on a FineWeb-Edu sample.

**Requirements:**
- Stream 200,000 documents from FineWeb-Edu `sample-10BT`
- Train `ByteLevelBPETokenizer` with `vocab_size=32000`, `min_frequency=2`
- Include special tokens: `["<|endoftext|>"]`
- Save tokenizer to `pretrain-50m/tokenizer/`
- Verify by encoding the sentence: `"SELECT id, created_at FROM events WHERE user_id = 42"` and printing the token IDs and decoded text

**Deliverable:** `pretrain-50m/tokenizer/vocab.json` and `merges.txt`. A verification printout showing the SQL sentence tokenized correctly.

**Acceptance criteria:** 
- vocab.json contains exactly 32,000 entries
- `<|endoftext|>` is in the vocabulary
- The SQL sentence roundtrips cleanly (encode then decode produces the original string)

**Hints:**
- A generator function is more memory-efficient than loading all texts into a list:
```python
def text_iterator(ds, n=200_000):
    for i, doc in enumerate(ds):
        if i >= n: break
        yield doc["text"]
tokenizer.train_from_iterator(text_iterator(ds), vocab_size=32000, ...)
```
- Training 200K docs takes about 5–15 minutes on Colab CPU

---

## Task 2 — Build the Data Pipeline

**Goal:** Tokenize FineWeb-Edu into a memory-mapped binary file.

**Requirements:**
- Write `pretrain-50m/data/prepare.py` that:
  - Streams FineWeb-Edu (sample-10BT)
  - Tokenizes each document with your BPE tokenizer
  - Appends the `<|endoftext|>` token after each document
  - Writes the result to `train.bin` and `val.bin` (90% / 10% split by document order)
  - Target: at least 100M tokens in `train.bin` (more is better for Week 21)
- Write `pretrain-50m/data/dataset.py` implementing `TokenDataset` (see Curriculum)
- Verify: `len(np.memmap("train.bin", dtype=np.uint16, mode="r"))` gives expected token count

**Deliverable:** `prepare.py`, `dataset.py`, `train.bin`, `val.bin`. Report token counts in `week-20-setup.md`.

**Acceptance criteria:**
- `train.bin` contains at least 100M tokens
- A single `__getitem__` call from `TokenDataset` returns tensors of shape `(block_size,)` each
- Tokens are dtype `int64` when returned by the Dataset (even though stored as `uint16`)

---

## Task 3 — Implement the 50M GPT Model

**Goal:** Write a GPT model that is fully yours (not a copy of nanoGPT).

**Requirements:**
- Implement `CausalSelfAttention` with:
  - `n_heads=12`, `d_model=768`
  - Causal masking via `F.scaled_dot_product_attention(q, k, v, is_causal=True)`
  - Attention dropout=0.0 (no dropout for pretraining at this scale)
- Implement `MLP` with GELU activation, `d_ff = 4 × d_model`
- Implement `TransformerBlock` with pre-LayerNorm (LN before attention and before MLP)
- Implement `GPT` with:
  - `n_layers=8`, `d_model=768`, `n_heads=12`, `vocab_size=32000`, `context_len=1024`
  - Weight tying between embedding and LM head
- Verify: `count_params(model)` returns between 50M and 65M

**Deliverable:** `pretrain-50m/model.py`

**Acceptance criteria:**
- `python -c "from model import GPT; m = GPT(); print(sum(p.numel() for p in m.parameters())/1e6, 'M params')"` produces between 50 and 65
- A forward pass with random input of shape `(2, 1024)` completes without error
- Output logits shape is `(2, 1024, 32000)`

---

## Task 4 — Write the Training Loop and Sanity Check

**Goal:** Wire up the model, data, Accelerate, and W&B. Run 200 steps to verify everything works.

**Requirements:**
- Write `pretrain-50m/train.py` with:
  - Accelerate integration (mixed precision bf16, gradient accumulation 4)
  - AdamW optimizer: lr=3e-4, weight decay=0.1, beta1=0.9, beta2=0.95
  - Cosine LR schedule with 100-step linear warmup
  - Log `train/loss`, `train/lr`, `train/tokens_seen` to W&B every 10 steps
  - Save checkpoint every 500 steps to `checkpoints/`
  - Evaluate val loss every 200 steps
- Run for 200 steps and verify:
  - Initial loss ≈ `log(32000)` ≈ 10.4
  - Loss after 200 steps < 7.0

**Deliverable:** `pretrain-50m/train.py`, W&B run link, `week-20-setup.md` with sanity check results.

GitHub commit: `week-20-pretrain-setup`

---

## Stretch Goals

- Add `torch.compile(model)` (requires PyTorch 2.0+) and measure speed improvement in tokens/sec
- Implement gradient clipping: `accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)`
- Profile memory usage with `torch.cuda.memory_summary()` and estimate how many tokens/sec you achieve
