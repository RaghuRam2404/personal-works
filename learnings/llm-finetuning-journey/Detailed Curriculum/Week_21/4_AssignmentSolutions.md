# Week 21 Assignment Solutions

## Task 1 — Key Snippet: Complete Training Loop

```python
import time, math, torch
from accelerate import Accelerator

accelerator = Accelerator(mixed_precision="bf16",
                          gradient_accumulation_steps=GRAD_ACCUM)
model, optimizer = accelerator.prepare(model, optimizer)
if COMPILE: model = torch.compile(model)

step = start_step
tokens_seen = step * BATCH_SIZE * GRAD_ACCUM * BLOCK_SIZE

for step in range(start_step, MAX_STEPS):
    model.train()
    t0 = time.time()

    with accelerator.accumulate(model):
        x, y = next(train_iter)
        x, y = x.to(accelerator.device), y.to(accelerator.device)
        logits, loss = model(x, y)
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            grad_norm = accelerator.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        optimizer.zero_grad()

    tokens_seen += BATCH_SIZE * GRAD_ACCUM * BLOCK_SIZE
    dt = time.time() - t0
    tps = (BATCH_SIZE * GRAD_ACCUM * BLOCK_SIZE) / dt

    if step % LOG_INTERVAL == 0:
        accelerator.log({
            "train/loss": loss.item(),
            "train/lr": get_lr(step),
            "train/grad_norm": grad_norm.item() if accelerator.sync_gradients else 0,
            "train/tokens_per_sec": tps,
            "train/tokens_seen": tokens_seen,
        }, step=step)

    if step % EVAL_INTERVAL == 0:
        val_loss = evaluate(model, val_loader, accelerator)
        accelerator.log({"val/loss": val_loss}, step=step)

    if step % SAVE_INTERVAL == 0:
        accelerator.save_state(f"checkpoints/step_{step:06d}")
```

**Common gotchas:**
- Not updating `tokens_seen` correctly — divide by `GRAD_ACCUM` if you only increment per accumulation step
- Logging inside the accumulation loop (before sync) — log only when `accelerator.sync_gradients` is True
- `torch.compile` with Colab causes a 2–5 minute wait on first forward pass — do not mistake this for a hang

---

## Task 3 — HuggingFace Upload Snippet

```python
from transformers import GPT2Config, GPT2LMHeadModel
import torch

# Map your config to GPT2Config
hf_config = GPT2Config(
    vocab_size=32000,
    n_positions=1024,
    n_embd=768,
    n_layer=8,
    n_head=12,
    resid_pdrop=0.0,
    embd_pdrop=0.0,
    attn_pdrop=0.0,
)

hf_model = GPT2LMHeadModel(hf_config)

# Load your weights — need to remap key names
ckpt = torch.load("checkpoints/best_model.pt", map_location="cpu")
# Key remapping depends on your naming convention
# e.g., "blocks.0.attn.qkv.weight" → "transformer.h.0.attn.c_attn.weight"
# This is the hard part — do it once carefully

hf_model.push_to_hub("<your-handle>/fineweb-50m-pretrain")
```

Alternative (simpler): save your model with your own `save_pretrained` using a custom config JSON, and document the loading procedure in the model card. Weight remapping to GPT2Config is tedious — accept either approach.

---

## Expected Training Metrics

| Metric | Expected Value |
|---|---|
| Initial val loss | ~10.4 |
| Val loss at 5K steps (~100M tokens) | 4.5–5.5 |
| Val loss at 15K steps (~500M tokens) | 3.5–4.2 |
| Val loss at 30K steps (~2B tokens) | 2.8–3.5 |
| Average tokens/sec | 40,000–70,000 |
| Total wall-clock time | 8–14 hours |

**If your val loss is above 5.0 at 15K steps:** check that you did not accidentally overfit to a tiny dataset (check your .bin file has >500M tokens).

**If your val loss is below 2.5 at 15K steps:** something is wrong — likely training and val data overlap, or eval is running on a tiny subset. Investigate.

**Red flags in the training log:**
- `grad_norm` is always exactly 1.0 → grad clipping fires every step → LR may be too high
- `tokens_per_sec` drops over time → memory leak (check for tensors accumulating in a list)
- `val/loss` is flat while `train/loss` drops → your val iterator is broken (not iterating)
