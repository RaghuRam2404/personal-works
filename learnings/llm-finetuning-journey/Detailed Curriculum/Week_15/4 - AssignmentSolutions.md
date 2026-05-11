# Week 15 Assignment Solutions

## Task 1 — Weight Loading from HuggingFace GPT-2

```python
@classmethod
def from_pretrained(cls, model_type):
    from transformers import GPT2LMHeadModel
    config_args = {
        'gpt2':        dict(n_layer=12, n_head=12, n_embd=768),
        'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
        'gpt2-large':  dict(n_layer=36, n_head=20, n_embd=1280),
        'gpt2-xl':     dict(n_layer=48, n_head=25, n_embd=1600),
    }[model_type]
    config_args.update(vocab_size=50257, block_size=1024)
    model = cls(GPTConfig(**config_args))
    sd = model.state_dict()
    sd_hf = GPT2LMHeadModel.from_pretrained(model_type).state_dict()
    # HF GPT2 uses Conv1D (transposed) for attention projections
    transposed_keys = ['attn.c_attn.weight', 'attn.c_proj.weight',
                       'mlp.c_fc.weight', 'mlp.c_proj.weight']
    for k in sd:
        if any(k.endswith(t) for t in transposed_keys):
            sd[k].copy_(sd_hf[k].T)
        else:
            sd[k].copy_(sd_hf[k])
    return model
```

The critical detail: HuggingFace GPT-2 uses `Conv1D` (from transformers) where weights are stored transposed relative to `nn.Linear`. Any `c_attn`, `c_proj`, `c_fc` weights must be `.T` when copying.

**Verification:** `model.generate(encode("Hello"), 20)` should produce readable English.

---

## Task 2 — Training Loop Key Snippet

```python
optimizer.zero_grad()
loss_accum = 0.0
for micro_step in range(grad_accum_steps):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        logits, loss = model(x, y)
    loss = loss / grad_accum_steps   # CRITICAL: normalize
    loss_accum += loss.detach()
    loss.backward()                   # accumulates gradients

# After all micro-steps:
norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
lr = get_lr(step)
for param_group in optimizer.param_groups:
    param_group['lr'] = lr
optimizer.step()
```

**AdamW with weight decay only on 2D tensors:**
```python
decay_params = [p for n, p in model.named_parameters()
                if p.dim() >= 2]
nodecay_params = [p for n, p in model.named_parameters()
                  if p.dim() < 2]
optimizer = torch.optim.AdamW([
    {'params': decay_params, 'weight_decay': 0.1},
    {'params': nodecay_params, 'weight_decay': 0.0},
], lr=max_lr, betas=(0.9, 0.95), eps=1e-8)
```

---

## Expected Training Numbers

| Step | Train Loss | Val Loss | HellaSwag | Tokens/sec (A100) |
|---|---|---|---|---|
| 0 | ~10.8 | ~10.8 | ~25% | — |
| 1000 | ~4.5 | ~4.6 | ~26% | ~300k |
| 5000 | ~3.6 | ~3.65 | ~27% | ~300k |
| 19073 (full epoch) | ~3.11 | ~3.11 | ~29.5% | ~300k |

Target: val loss < 3.27 at any point during training (within 5% of 3.11 is the soft goal; hitting exactly 3.11 requires the full training run).

---

## Common Gotchas

- Not dividing loss by `grad_accum_steps` → gradients are 32x too large → NaN in first optimizer step.
- Clipping gradients inside the micro-step loop → clips partial gradients incorrectly; clip once after the full accumulation loop.
- Using `torch.float16` instead of `torch.bfloat16` → numerical instability on some operations; use bfloat16 on A100.
- Forgetting to call `optimizer.zero_grad()` at the start of each logical batch (not each micro-step) → gradients from multiple logical batches accumulate.
- `weight_decay=0.1` applied to biases and LayerNorm parameters → hurts convergence; exclude 1D tensors.

---

## How to Verify You Did It Right

1. Step 0 loss ≈ -log(1/50257) ≈ 10.82. If it's very different, your loss computation has a bug.
2. After loading pretrained weights, val loss ≈ 3.11 (within 0.05). This confirms weight loading is correct.
3. HellaSwag accuracy starts at 25% (random for 4-way) at step 0.
4. `tokens_per_second` on A100 with bfloat16: ~250,000–350,000 tok/s. If < 100k, check that autocast is active.
5. W&B `grad_norm` should be around 0.8–1.5 after warmup. If it's consistently hitting the 1.0 clip, your LR may be too high.
