# Week 15 Assignment — Reproducing GPT-2 124M

## Setup Checklist

- [ ] Buy Colab Pro ($10, one month). Use the A100 runtime when available. This is a required spend.
- [ ] Install: `pip install tiktoken datasets wandb`
- [ ] W&B project `week-15-gpt2-repro`
- [ ] GitHub branch `week-15-gpt2-repro`
- [ ] 20GB+ free disk on Colab (for FineWeb-Edu or OpenWebText download)

---

## Task 1 — Implement GPT-2 124M Architecture (Code Along)

**Goal:** Type out the full GPT-2 124M architecture from Karpathy's video.

**Requirements:**
- GPT-2 config: `n_layer=12, n_head=12, n_embd=768, vocab_size=50257, block_size=1024`
- Use tiktoken GPT-2 tokenizer: `import tiktoken; enc = tiktoken.get_encoding("gpt2")`
- Implement `CausalSelfAttention` using `F.scaled_dot_product_attention(q, k, v, is_causal=True)` — this is Flash Attention-compatible
- GELU activation (not SwiGLU — GPT-2 uses GELU)
- Pre-LN (Karpathy confirms this is what GPT-2 actually uses)
- Weight tying: `lm_head.weight = transformer.wte.weight`
- `@classmethod from_pretrained(cls, model_type)` that loads weights from HuggingFace `gpt2` checkpoint
  - Load `openai-community/gpt2` via `from transformers import GPT2LMHeadModel`
  - Copy weights manually with correct transpositions (Conv1D vs. Linear convention difference)

**Verification:** Load HF GPT-2 weights into your model. Generate text from "Hello, I'm a language model," — if generation is coherent, weights transferred correctly.

**Deliverable:** `train_gpt2.py` containing the model class and weight loading.

---

## Task 2 — Implement the Training Loop

**Goal:** Full production-quality training loop with mixed precision, gradient accumulation, and HellaSwag evaluation.

**Requirements:**
- Data: download OpenWebText or FineWeb-Edu (10B tokens shard)
  - Tokenize with tiktoken, save as numpy uint16 array: `tokens.astype(np.uint16)`
  - `DataLoader` that randomly samples `(B, T)` chunks from this array
- Mixed precision: `torch.autocast(device_type='cuda', dtype=torch.bfloat16)`
- Gradient accumulation:
  - `total_batch_size = 524288` (0.5M tokens)
  - `B = 16, T = 1024` per micro-batch on A100 (adjust for your GPU)
  - `grad_accum_steps = total_batch_size // (B * T)` = 32
  - Divide loss by `grad_accum_steps` before `loss.backward()`
- Cosine LR schedule with warmup (see Curriculum.md formula): `max_lr=6e-4, min_lr=6e-5, warmup_steps=715`
- Gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` before optimizer step
- Optimizer: `AdamW(betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)` — apply weight decay only to 2D tensors (not biases or 1D LayerNorm params)
- HellaSwag eval: every 250 steps, compute accuracy on the HellaSwag validation set
- Val loss: every 250 steps, compute val loss on 20 batches of held-out data

**Log to W&B:**
- Every step: `train_loss`, `lr`, `grad_norm`, `tokens_per_second`
- Every 250 steps: `val_loss`, `hellaswag_acc`

**Deliverable:** `train_gpt2.py` updated with training loop. W&B run `week-15-gpt2-repro`.

---

## Task 3 — Run the Actual Training

**Goal:** Train to a val loss ≤ 3.27 on OpenWebText.

**Requirements:**
- Run on Colab Pro A100 for as long as needed (Karpathy achieves ~3.11 in ~4 hours at full 124M scale)
- If compute is limited: run for at least 5000 steps and confirm the loss is decreasing on schedule
- Save a checkpoint every 1000 steps: `torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step}, 'ckpt_{step}.pt')`
- Generate and print sample text every 500 steps

**Acceptance criteria:**
- Val loss ≤ 3.27 (within 5% of GPT-2 124M's published 3.11)
- W&B run shows smooth loss curve with cosine decay visible
- HellaSwag acc ≥ 28% (vs. GPT-2's 29.5%)
- Generated text at the end of training is coherent English

**Deliverable:** GitHub commit `week-15-gpt2-repro` containing:
- `train_gpt2.py`
- `sample_outputs.txt` (5 generated samples at end of training)
- Link to W&B run in `README.md`

---

## Task 4 — (No-GPU Option) Load and Evaluate Pretrained Weights

If Colab Pro is unavailable this week (buy it next week), complete this task instead:

**Requirements:**
- Load HuggingFace GPT-2 124M weights into your model using your `from_pretrained` classmethod
- Compute val loss on 50 OpenWebText validation batches
- Generate 5 samples from the prompt "SELECT name FROM"
- Confirm val loss matches published GPT-2 val loss (~3.11)

**This does not substitute for Task 3.** You must still train from scratch. Use this as a debugging tool.

---

## Stretch Goals

- Implement the HellaSwag evaluation from scratch (read the data format, compute log-probs per candidate).
- Add `torch.compile(model)` (PyTorch 2.0). Measure speedup vs. non-compiled (typically 10–30% faster).
- Try training the same model architecture with your Week 12 improvements (RMSNorm, SwiGLU, RoPE instead of GPT-2's original config). Is the modernized version more efficient at the same step count?
