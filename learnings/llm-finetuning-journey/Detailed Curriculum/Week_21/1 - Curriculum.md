# Week 21 — Run 50M LM Pretraining

## Learning Objectives

By the end of this week, you will be able to:

- Execute a multi-hour pretraining run with proper monitoring and checkpointing
- Identify divergence, instability, and healthy training from W&B loss curves
- Debug common training failure modes (loss spike, NaN loss, OOM) under time pressure
- Interpret tokens/sec throughput and estimate remaining training time
- Recover from training interruptions by resuming from checkpoints

---

## Concepts

### What This Week Is About

This is the most hands-on week of Phase 3. You are running a real training job — not a tutorial, not a toy, but a full pretraining run targeting ~2 billion tokens across approximately 24 hours of A100 GPU time (spread across the week).

The learning objective is not to produce a great language model. At 50M parameters, your model will be significantly worse than GPT-2. The objective is to experience the full training loop end-to-end: setup, launch, monitoring, debugging, checkpointing, and graceful termination.

Everything in this week is preparation for your Week 22 evaluation and for the fine-tuning you will do in Phases 4–6.

### Planning Your Training Run

Before you launch, calculate your training budget:

```
Target tokens: 2B
Train .bin size: ~1B tokens (from Week 20)
→ need ~2 epochs over the dataset

batch_size = 16, block_size = 1024
→ tokens per step = 16 × 1024 = 16,384
→ steps for 2B tokens = 2B / 16384 ≈ 122,000 steps

Effective batch size with gradient_accumulation=4: 4 × 16 × 1024 = 65,536 tokens/step
→ steps = 2B / 65536 ≈ 30,500 gradient update steps

At 50K tokens/sec: 2B / 50000 = 40,000 sec ≈ 11 hours
```

Plan to run two Colab Pro A100 sessions of ~5–6 hours each across the week. Save a checkpoint after each session.

### Hyperparameter Reference

Use these proven settings for a 56M model on FineWeb-Edu:

```python
# Architecture (from Week 20)
n_layers    = 8
d_model     = 768
n_heads     = 12
vocab_size  = 32_000
context_len = 1024

# Training
batch_size              = 16       # per-GPU
gradient_accumulation   = 4        # effective batch = 65536 tokens
max_lr                  = 3e-4
min_lr                  = 3e-5
warmup_steps            = 200      # ~13M tokens warmup
max_steps               = 30_500   # ~2B tokens
weight_decay            = 0.1
beta1, beta2            = 0.9, 0.95
grad_clip               = 1.0
mixed_precision         = "bf16"   # prefer bf16 over fp16 on A100

# Logging and checkpointing
log_interval    = 10       # log every 10 steps
eval_interval   = 500      # evaluate val loss every 500 steps
save_interval   = 2000     # save checkpoint every 2000 steps
```

**Why bf16 instead of fp16?**
A100 has first-class support for bfloat16 (bf16). BF16 has a wider exponent range (8 bits vs 5 bits for fp16), reducing overflow/underflow issues. With bf16, you typically do not need a GradScaler.

### Reading Your Training Curves

W&B will show you the following signals. Know what each means:

**Healthy training curve:**
- Train loss decreases steadily and smoothly
- Val loss tracks train loss with a small gap
- Gradient norm stays below 2.0 (with clip=1.0 it will be clipped frequently early on)
- Tokens/sec is stable (not dropping over time)

**Loss spike:** A sudden upward jump in train loss, typically followed by recovery. Causes:
- Bad batch (very long document, unusual characters)
- LR too high causing gradient explosion
- NaN in gradient (propagated from a numerical issue)

If a spike occurs and recovers → log it and continue. If loss spikes and does not recover → reload last checkpoint, reduce LR by 2×, resume.

**NaN loss:** Gradient or loss becomes NaN. Causes:
- Overflow in fp16 (switch to bf16 or add GradScaler)
- Unstable softmax (too large logits) — fix with output scaling
- Bad data (token ID out of range → embedding lookup returns garbage)

**Loss plateau:** Loss stops decreasing after rapid initial drop. Causes:
- Dataset is too small and model has memorized it (over many epochs)
- LR has decayed too aggressively
- Grad accumulation not working correctly

### Checkpoint Management

Save two types of checkpoints:

1. **Regular checkpoints** (every 2000 steps): full model state + optimizer state
2. **Best checkpoint** (based on val loss): overwrite a single `best_model.pt`

```python
# Checkpoint save
checkpoint = {
    'step': step,
    'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'val_loss': val_loss,
    'config': config,
}
torch.save(checkpoint, f'checkpoints/step_{step:06d}.pt')

# Checkpoint resume
if resume_from is not None:
    ckpt = torch.load(resume_from)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    start_step = ckpt['step'] + 1
```

### Monitoring Throughput

Add throughput tracking to your training loop:

```python
import time

t0 = time.time()
# ... training step ...
dt = time.time() - t0
tokens_per_sec = (batch_size * gradient_accumulation * block_size) / dt
```

Expected throughput on Colab A100 for a 56M model:
- Without `torch.compile`: 40,000–60,000 tokens/sec
- With `torch.compile`: 55,000–80,000 tokens/sec

If you see < 20,000 tokens/sec, something is wrong: check that mixed precision is active, that you are not accidentally running on CPU, and that `block_size=1024` (not 512).

### When Colab Disconnects

Colab Pro sessions can be interrupted. Design your training loop to:
1. Save every 2000 steps (at 50K tok/sec, this is every ~40 seconds — fast enough)
2. Use step number in checkpoint filenames to find the latest
3. Restore the DataLoader position using the step count to skip already-seen data:

```python
# Skip already-processed data on resume
if start_step > 0:
    # Simple approach: set random seed based on step, or just continue from
    # the beginning of the dataset (2 epochs is fine for 1B token dataset)
    pass
```

For a 1B-token dataset with 30K steps, you are doing ~2 epochs — restarting from the beginning after a disconnect introduces ~1 extra epoch, which is acceptable.

### What "2B Tokens" Means for Your Model

Chinchilla optimal for 56M params is ~1.1B tokens. You are targeting 2B — about 1.8× over-Chinchilla. This is fine:
- Your model is for learning, not deployment
- The extra tokens improve convergence in the absence of learning rate tuning perfection
- You will likely not be able to distinguish 1.1B-token vs. 2B-token results in perplexity

Accept whatever your Colab budget allows. The important thing is to get to at least 1B tokens for Week 22 evaluation.

---

## Connections

**Prior week (20):** All of Week 20's setup is the prerequisite for this week.

**Week 22:** The checkpoint from this run is your Week 22 starting point. Save it carefully.

**Phase 4 onwards:** This experience of reading training curves, debugging instability, and managing checkpoints is exactly what you will apply when fine-tuning 7B models.

---

## Common Misconceptions

- **"I should try different hyperparameters if loss is slow."** Do not change hyperparameters mid-run unless you have a catastrophic failure (NaN, irrecoverable spike). Let the run complete. You can analyze and adjust in a second run.
- **"Colab Pro A100 gives me 100% GPU time."** No — Colab Pro A100 gives you priority access, but you may be disconnected after 12–24 hours. Have checkpointing working before starting.
- **"I should increase batch size to train faster."** At 56M params on A100, the current batch size is likely already close to optimal for memory bandwidth. Larger batch sizes improve throughput only until memory is saturated.
- **"My val loss increasing slightly means overfitting."** At early steps, val loss can lag train loss temporarily. A small gap (<0.3) is expected and not concerning.

---

## Time Allocation (6–8 hrs)

- 1h: Pre-launch checklist (re-run sanity check from Week 20, verify checkpointing works)
- 5–6h: Active training on Colab A100 (can be background — check W&B every 30 min)
- 1h: Debug any issues, log findings in `journal.md`
- 0.5h: Commit checkpoint link and journal entry
