# Week 5 — Assignment

## Setup Checklist

- [ ] Your Week 3 CIFAR-10 CNN code is available and working in Colab.
- [ ] W&B project `week-05-optimization` created.
- [ ] Colab with T4 GPU (needed for AMP benchmarking).

---

## Task 1 — Implement Warmup + Cosine Schedule from Scratch

**Goal:** Understand LR schedules by deriving and implementing them without using PyTorch's scheduler helpers.

**Requirements:**
- Create `week_05/lr_schedule.py`.
- Implement a function `get_lr(step, warmup_steps, total_steps, max_lr, min_lr=0.0)` that returns the learning rate for a given step number, using:
  - Linear warmup for `step < warmup_steps`.
  - Cosine decay for `step >= warmup_steps`.
  - Formula for cosine phase: `lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * (step - warmup_steps) / (total_steps - warmup_steps)))`.
- Plot the LR schedule for `max_lr=3e-4`, `warmup_steps=200`, `total_steps=2000` using matplotlib. Save as `week_05/lr_curve.png`.
- Verify: at `step=0`, LR should be 0 (or near 0). At `step=warmup_steps`, LR should equal `max_lr`. At `step=total_steps`, LR should equal `min_lr`.
- Integrate this function into a training loop by manually setting `optimizer.param_groups[0]['lr'] = get_lr(step, ...)` each step.

**Deliverable:** `week_05/lr_schedule.py` + `week_05/lr_curve.png`.

---

## Task 2 — CIFAR-10 Optimization Comparison

**Goal:** Demonstrate quantitatively that optimizer + schedule choice matters.

**Requirements:**
- Run three training configurations of your Week 3 CIFAR-10 CNN, each for 20 epochs on Colab T4. Log all runs to W&B project `week-05-optimization`:

  | Run name | Optimizer | LR | Schedule | Grad Clip | AMP |
  |---|---|---|---|---|---|
  | `baseline` | SGD (momentum=0.9, wd=5e-4) | 0.1 | None (constant) | No | No |
  | `adamw-cosine` | AdamW (β1=0.9, β2=0.999, wd=0.01) | 3e-4 | Cosine (warmup 200 steps) | 1.0 | No |
  | `adamw-cosine-amp` | AdamW (same) | 3e-4 | Cosine (warmup 200 steps) | 1.0 | Yes (`autocast`) |

- For each run, log every step: `train/loss`, `train/acc`, `val/loss`, `val/acc`, `train/lr`, `train/grad_norm` (before clipping).
- After training, generate a W&B comparison report showing all three runs on the same axes. Save a screenshot of the report as `week_05/wb_comparison.png`.
- In `week_05/comparison_notes.md`, write 2–3 paragraphs explaining: which run converges faster? Which achieves the best final accuracy? Did AMP cause any accuracy change? Estimate the speedup from AMP (compare epoch time).

**Deliverable:** Three W&B runs, comparison report screenshot, `week_05/comparison_notes.md`. Commit message: `week-05-optimization`.

**Hints:**
- Log gradient norm before clipping: `grad_norm = torch.nn.utils.get_total_norm(model.parameters())` — actually, compute it before `clip_grad_norm_` by summing squared norms: `total_norm = sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None) ** 0.5`.
- For AMP with GradScaler, the optimizer step is replaced by `scaler.step(optimizer); scaler.update()`.
- Do not change the architecture between runs — same model, different optimizer/schedule only.

---

## Task 3 — Optimizer Internals: Implement Adam from Scratch

**Goal:** Understand Adam's update rule by implementing it yourself.

**Requirements:**
- Create `week_05/adam_scratch.py`.
- Implement an `AdamOptimizer` class with:
  - `__init__(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)`.
  - `step()` method that applies the AdamW update (decoupled weight decay, not Adam's L2 version).
  - Maintains per-parameter `m` (first moment) and `v` (second moment) state.
  - Handles bias correction correctly.
- Train your MLP from Week 2 on `make_moons` using your `AdamOptimizer`. Verify that the loss curve approximately matches `torch.optim.AdamW` with the same hyperparameters.

**Deliverable:** `week_05/adam_scratch.py`.

**Hints:**
- Store optimizer state in a list of dicts, one per parameter. `param._optimizer_state = {'m': torch.zeros_like(param.data), 'v': torch.zeros_like(param.data), 't': 0}`. Or maintain a parallel list in the optimizer itself.
- AdamW update: after the Adam step, apply `param.data -= lr * weight_decay * param.data` (the weight decay is applied to the parameter data, not the gradient).

---

## Stretch Goals

- Implement a learning rate range test (LR Finder): start with LR 1e-8, multiply by a factor each step, and log the loss vs. LR. Find the point where loss starts increasing — that is your max safe LR. This technique (from Smith 2018) is widely used in practice.
- Read the [Lion optimizer paper](https://arxiv.org/abs/2302.06675) (2023). In 1 paragraph, explain the key difference between Lion and AdamW and why Lion might be useful.
