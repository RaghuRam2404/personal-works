# Week 15 — From-Scratch GPT-2 124M Reproduction (Karpathy)

## Learning Objectives

By end of this week, you will be able to:

- Reproduce GPT-2 124M training following Karpathy's methodology
- Implement mixed-precision training (torch.autocast with bfloat16)
- Implement gradient accumulation to simulate large batch sizes on limited hardware
- Understand Flash Attention and use `F.scaled_dot_product_attention` as a drop-in
- Explain what HellaSwag evaluation measures and why it's used during pretraining
- Achieve val loss within 5% of GPT-2 124M's published val loss on OpenWebText

## Time Allocation (6–8 hrs)

| Activity | Time |
|---|---|
| Watch Karpathy's full video (4h01m) — code along | 5 hrs |
| Run actual training on Colab Pro (A100) | 1 hr setup + monitor |
| Review W&B plots, compare to published numbers | 1 hr |
| Commit and document | 0.5 hrs |

This is the most time-intensive coding week in Phase 2. Spread the video across the full week. Do not rush it.

---

## Concepts

### Why This Week Matters

Every previous week has been building toward this: you understand attention (Weeks 9–10), you built a decoder-only model (Week 11), you added modern architecture components (Week 12), you implemented KV cache and sampling (Week 13), you read the LLaMA codebase (Week 14). Now you reproduce a real, published model from scratch.

This is not a toy. GPT-2 124M is a real model that OpenAI published in 2019. Karpathy's video shows how to reproduce its val loss on OpenWebText — if you achieve this, you have proven you understand every detail of the modern LLM training stack.

### Architecture: GPT-2 vs. Your nanoGPT

GPT-2 124M uses:
- `n_layer=12, n_head=12, n_embd=768, vocab_size=50257, block_size=1024`
- Post-LN in the original paper, but GPT-2's actual code uses **Pre-LN** (this was confirmed by Karpathy — the paper description is misleading)
- GELU activation (not SwiGLU — this is the original GPT-2, not LLaMA-style)
- No RoPE — learned absolute position embeddings
- Weight tying (embedding and LM head share weights)

This is essentially your Week 11 nanoGPT with larger hyperparameters and the tokenizer changed to tiktoken (GPT-2's BPE tokenizer, 50257 vocab).

### Mixed-Precision Training (bfloat16)

Training large models in FP32 requires 4 bytes per parameter. FP16 uses 2 bytes but has limited range (overflow risk). BFloat16 (bfloat16, `torch.bfloat16`) uses 2 bytes but with the same exponent range as FP32 — just lower precision in the mantissa. For neural network training, this trade-off is almost always acceptable.

In PyTorch:
```python
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    logits, loss = model(x, y)
# gradients computed in FP32 for stability
loss.backward()
```

`torch.autocast` automatically casts forward pass operations to bfloat16 while keeping sensitive operations (layer norm, softmax) in FP32. The optimizer states (AdamW) remain in FP32. This is called "mixed precision" — the forward pass is BF16, the weight updates are FP32.

Speedup: on A100/H100, bfloat16 is 2–3x faster than FP32 due to tensor core utilization.

### Gradient Accumulation

GPT-2 was trained with batch size 0.5M tokens. On a single A100 with 40GB VRAM, you can fit `batch_size × seq_len = B × T` tokens where `B × T ≤ ~100k` at 124M parameters. To simulate a large batch without the memory:

```python
# Simulate total_batch_size = 524288 tokens with B=4, T=1024 per step
grad_accum_steps = total_batch_size // (B * T)  # = 128 steps

optimizer.zero_grad()
for micro_step in range(grad_accum_steps):
    x, y = get_batch()
    with torch.autocast(...):
        logits, loss = model(x, y)
    loss = loss / grad_accum_steps  # normalize
    loss.backward()  # accumulate gradients
optimizer.step()
```

Each micro-step adds to the gradient. After `grad_accum_steps` micro-steps, the accumulated gradient is equivalent to a single step with the full large batch. The LR schedule, gradient clipping, and optimizer step all happen once per "logical" batch.

### Flash Attention

The naive attention implementation computes a `[T × T]` matrix in HBM (GPU memory), which is slow and memory-intensive. Flash Attention (Dao et al., 2022) computes attention in tiles, keeping the working memory in fast SRAM (L1/L2 cache). No `[T × T]` matrix is ever materialized in HBM.

In PyTorch 2.0+:
```python
# Drop-in replacement for manual attention (uses Flash Attention if available)
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

Speedup over naive attention: 2–4x on A100. Memory: O(T) instead of O(T^2). For long contexts this is critical; for GPT-2 at T=1024, it's a meaningful but not critical speedup.

### HellaSwag Evaluation

During pretraining, you can't use downstream task metrics (the model isn't instruction-tuned). Instead, Karpathy evaluates on HellaSwag — a commonsense NLI benchmark. Given 4 candidate sentence completions, pick the most plausible one. For a language model, this is framed as: compute perplexity of each candidate under the LM; pick the one with the lowest perplexity.

GPT-2 124M achieves ~29.5% HellaSwag accuracy (vs. random=25%). As training progresses, you can track HellaSwag to verify your model is developing real language understanding, not just memorizing.

### Learning Rate Schedule

GPT-2 reproduction uses:
- **Warmup:** Linear LR increase from 0 to `max_lr` over `warmup_steps` (typically 715 steps = first 10M tokens)
- **Cosine decay:** LR decays from `max_lr` to `min_lr = 0.1 × max_lr` following a cosine curve
- **Max LR:** 6e-4 (for 124M model)
- **Min LR:** 6e-5

```python
def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
```

### What "Within 5% of Published Val Loss" Means

GPT-2 124M achieves approximately 3.11 val loss on OpenWebText (this is the community-measured number since OpenAI only published the model, not the val loss). Karpathy achieves ~3.11 in his video. Your target: val loss < 3.27 (3.11 × 1.05 = 5% tolerance).

Training GPT-2 124M to this quality requires approximately 10 billion tokens (roughly Karpathy's training run) on an A100. On Colab Pro's A100, this takes approximately 4–6 hours. This is where your first real compute spend happens ($10 for Colab Pro for one month).

## Connections

**Building on:** Everything in Phase 2. This is the capstone.

**Used in:** Week 16 (Phase Gate — you must have done this to pass), Phase 3 (pretraining mechanics), Phase 4 (fine-tuning — you'll fine-tune a model you understand intimately).

## Common Misconceptions / Pitfalls

- **"I can train GPT-2 124M to full quality on free Colab."** No — the free tier has time limits and uses T4 (slower). Use Colab Pro with A100 for the actual long run.
- **"BF16 training is equivalent to FP16."** BF16 has more numerical range (good), less mantissa precision (usually fine for training). On older hardware (pre-Ampere), BF16 is not available — use FP16 with a GradScaler instead.
- **Gradient accumulation and gradient clipping order.** Always clip AFTER accumulation is complete, just before `optimizer.step()`. Clipping during accumulation clips partial gradients, which is wrong.
- **Forgetting to normalize loss by `grad_accum_steps`.** If you don't divide loss by `grad_accum_steps`, accumulated gradients are `grad_accum_steps` times too large, causing instability or incorrect effective LR.
- **Confusing train loss and val loss.** Train loss is computed over a single micro-batch with dropout active. Val loss is computed over a fixed validation set with `model.eval()`. These should track together but won't be identical.
