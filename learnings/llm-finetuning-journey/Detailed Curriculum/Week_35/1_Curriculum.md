# Week 35 — Hyperparameter Tuning for SFT/LoRA

## Learning Objectives

By the end of this week, you will be able to:

- Articulate what each major SFT/LoRA hyperparameter controls and how it affects loss curves
- Run a W&B sweep over LR, rank, and alpha on a 1K example subset
- Interpret sweep results to choose the best hyperparameter configuration for your 15K dataset (Week 38)
- Understand the interaction between learning rate, batch size, and gradient accumulation
- Apply practical heuristics for hyperparameter selection without exhaustive search

---

## Concepts

### 1. The Hyperparameters That Matter Most

For LoRA/QLoRA SFT, roughly ranked by impact:

1. **Learning rate (LR)** — single highest-impact hyperparameter
2. **LoRA rank (r)** — controls model capacity
3. **Number of epochs** — controls how long you train (overfitting risk)
4. **Effective batch size** — interacts with LR
5. **LoRA alpha** — scaling of the adapter output
6. **LR scheduler type** — how LR decays over time
7. **Warmup ratio** — prevents instability at the start
8. **Packing** — throughput optimization (minimal effect on quality)

### 2. Learning Rate

The most critical hyperparameter. For LoRA on 7B models, the typical range is `1e-5` to `5e-4`:

- **Too low (< 1e-5):** Adapter parameters update too slowly; model barely moves from base behavior. Loss decreases extremely slowly.
- **Sweet spot (2e-4 to 1e-4 for LoRA):** Loss decreases steadily without instability. Most practitioners find 2e-4 with cosine decay works well for rank 16 LoRA.
- **Too high (> 5e-4):** Loss spike at step ~100, possibly NaN. Gradient explosion overrides the LoRA scaling benefit.

For full SFT (no LoRA), optimal LR is much lower — typically `1e-5` to `5e-5`. The LoRA adapter is more forgiving of higher LR because the update magnitude is bounded by `alpha/r`.

**LR and rank interaction:** With `alpha = 2r` (scaling = 2), the effective LR for adapter outputs is `lr × alpha / r = lr × 2`. If you double the rank without changing alpha, you halve the scaling — effectively halving the influence of the adapter per gradient step. Keep `alpha = 2r` and tune LR independently.

### 3. Batch Size and Gradient Accumulation

Effective batch size = `per_device_train_batch_size × gradient_accumulation_steps × num_gpus`.

Larger effective batch size:
- More stable gradient estimates (less noise per step)
- Typically allows a slightly higher peak LR (linear scaling rule)
- Fewer steps per epoch for the same data

For SQL fine-tuning with 5K examples: effective batch 16 (4 × 4) is a solid default. At 15K examples (Week 38), you can increase to 32 or 64 if memory allows.

**The linear scaling rule:** If you double batch size, multiply LR by sqrt(2) (square root scaling) or 2 (linear scaling). The square root scaling is more conservative and typically safer for LLMs.

### 4. Epochs vs. Steps

For SFT on small datasets:
- 1K examples → 1–2 epochs maximum (risk of memorization)
- 5K examples → 2–3 epochs
- 15K examples → 2 epochs (with early stopping on eval loss)
- 100K+ examples → 1 epoch often sufficient

Watch for the gap between train and eval loss. If train loss continues decreasing while eval loss plateaus or rises, you have entered the overfitting regime. Stop training.

Early stopping: in `SFTConfig`, set `load_best_model_at_end=True` and `metric_for_best_model="eval_loss"`. This saves and reloads the checkpoint from the best eval loss seen during training.

### 5. LR Scheduler Types

| Scheduler | Shape | Best for |
|---|---|---|
| `cosine` | Smooth decay to near-0 | Standard choice; good for 1–5 epochs |
| `linear` | Linear decay to 0 | Simple; works well but can terminate training too aggressively |
| `constant` | No decay | Useful for very short runs; often suboptimal for longer |
| `cosine_with_restarts` | Periodic cosine | For long training with cyclic learning |

For your SQL fine-tuning: `cosine` is the default and works well. Use `warmup_ratio=0.05–0.1` to avoid instability in the first 5–10% of steps.

### 6. Packing

`packing=True` concatenates short examples for GPU efficiency. For SQL examples (typically 100–300 tokens), packing with `max_seq_length=512` achieves a packing ratio of 2–4x, dramatically improving throughput.

**Quality effect:** Packing introduces cross-example attention unless you use attention masking (not the default in SFTTrainer). In practice, the quality impact is small because the model learns to use the `<|im_end|>` token as a strong boundary signal. If you observe quality degradation, try `packing=False` as a diagnostic.

### 7. Running a W&B Sweep

W&B sweeps allow systematic hyperparameter search using Bayesian optimization or grid search:

```python
sweep_config = {
    "method": "grid",
    "metric": {"name": "eval/loss", "goal": "minimize"},
    "parameters": {
        "learning_rate": {"values": [1e-5, 5e-5, 2e-4]},
        "lora_rank": {"values": [16, 32]},
        "lora_alpha": {"values": [16, 32, 64]},
    }
}
sweep_id = wandb.sweep(sweep_config, project="week-35-hp-sweep")
wandb.agent(sweep_id, train_function, count=12)  # 3 × 2 × 2 = 12 runs
```

For this week: run on a 1K subset to keep each run short (5–10 minutes). The goal is not to find the perfect hyperparameter — the 1K subset is too small to reliably predict 15K performance — but to understand the sensitivity of the loss curve to each hyperparameter.

### 8. Practical Rules (Raschka's Empirical Findings)

- Fix: `alpha = rank` or `alpha = 2 * rank`. Do not tune alpha independently.
- Fix: `target_modules = all linear layers` (from Week 31 analysis).
- Tune: LR (most impactful), rank (second), epochs (via early stopping).
- For SQL task: LR = 2e-4 with rank 16 is a good starting point for 15K examples.
- Dropout: 0 (Unsloth) or 0.05 (vanilla peft). Do not tune unless you observe clear overfitting.

---

## Connections

**Builds on:** Week 31 rank sweep; Week 34 Unsloth setup. This week applies systematic tuning to your training setup.

**Needed for:** Week 38 (you will use the best hyperparameters found here for the 15K sprint).

---

## Common Misconceptions / Pitfalls

- **"Sweeping all hyperparameters simultaneously."** Sweeping 5+ hyperparameters at once with 3 values each = 243 runs. Run budget: 1 week on Colab = ~10 runs max. Fix: tune LR first, then rank, then others if time permits.
- **"Sweep on the full dataset."** Sweep on a 1K subset for speed, then verify the best config on 5K. The ordering of hyperparameter importance holds even if absolute loss values differ.
- **"Early stopping is always better."** If your eval set is too small (<200 examples), eval loss is noisy — early stopping may stop training based on noise. Use an eval set of at least 200–500 examples.
- **"Higher LR is always better because training is faster."** High LR can cause loss spikes that recover only partially, leaving the model at a worse local minimum than a lower LR run.

---

## Time Allocation (6–8 hrs)

| Activity | Time |
|---|---|
| Read Sebastian Raschka's LoRA insights article (fully) | 1.5h |
| Design sweep: choose which hyperparameters to sweep, define config | 30m |
| Write W&B sweep script | 1h |
| Run sweep (12 runs × 5–10 min each on 1K subset) | 2h |
| Analyze results, write recommendation | 1.5h |
| Commit to GitHub | 30m |
