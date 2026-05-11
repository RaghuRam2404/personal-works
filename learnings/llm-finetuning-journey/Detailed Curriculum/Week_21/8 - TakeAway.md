# Week 21 TakeAway — Running Pretraining

**One-liner:** Plan the token budget, checkpoint every 2000 steps, read loss curves critically, never change hyperparameters mid-run unless catastrophic.

---

## Training Budget Formula

```python
tokens_per_step = batch_size * grad_accum * block_size
total_steps     = target_tokens / tokens_per_step
wall_clock_hrs  = target_tokens / tokens_per_sec / 3600

# Example: 2B tokens at 50K tok/sec = 11 hours
```

---

## Hyperparameter Reference (56M model)

| Parameter | Value |
|---|---|
| max_lr | 3e-4 |
| min_lr | 3e-5 |
| warmup_steps | 200 |
| beta1, beta2 | 0.9, 0.95 |
| weight_decay | 0.1 |
| grad_clip | 1.0 |
| mixed_precision | bf16 |
| batch_size (per GPU) | 16 |
| grad_accum | 4 |
| effective batch (tokens) | 65,536 |

---

## Key Code Pattern — Checkpoint

```python
# Save (every 2000 steps)
torch.save({
    'step': step,
    'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'val_loss': val_loss,
}, f'checkpoints/step_{step:06d}.pt')

# Resume
ckpt = torch.load('checkpoints/step_006000.pt')
model.load_state_dict(ckpt['model_state'])
optimizer.load_state_dict(ckpt['optimizer_state'])
start_step = ckpt['step'] + 1
```

---

## Reading Training Curves

| Signal | Healthy | Problem |
|---|---|---|
| Train loss | Steady decrease | Plateau, spike, NaN |
| Val - train gap | < 0.3 | > 1.0 → overfitting |
| Grad norm | 0.1–2.0 | Clips every step → LR too high |
| Tokens/sec | Stable | Dropping → memory leak |

---

## Decision Rules

- Loss spike and recovers in < 200 steps → log and continue
- Loss spike and does not recover → reload last checkpoint, reduce LR by 2×
- NaN loss → check for uint16 token ID overflow; switch to bf16; check embedding range
- Clipping every step after warmup → reduce max_lr by 30%
- Tokens/sec below 20K on A100 → check mixed precision is active

---

## Numbers to Remember

| Metric | Value |
|---|---|
| Expected val loss (56M, 2B tokens) | 2.8–3.5 |
| Expected perplexity | 16–33 |
| Throughput (56M, A100, bf16) | 40K–70K tok/sec |
| Save overhead | negligible (<1 sec at 2000 step intervals) |
