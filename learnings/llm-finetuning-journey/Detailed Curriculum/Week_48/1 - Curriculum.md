# Week 48 — GRPO Sprint Week 2: Run It

## Learning Objectives

By the end of this week, you will be able to:

- Run a complete GRPO training job on RunPod A100 using Unsloth
- Monitor GRPO-specific metrics (mean reward, reward std, group advantages) in W&B
- Diagnose and fix GRPO training issues in real time
- Evaluate the resulting v3 model against v1 and v2 on your SQL benchmark
- Produce a trained `postgres-sqlcoder-7b-v3-grpo` that outperforms v2-dpo on complex queries

## This Week's Focus: Execution

You have spent Week 47 designing the reward function. This week you run it. The training session itself will be 5–10 hours on a RunPod A100 (roughly $10 at current rates). Plan your time accordingly: set up the run, monitor it, and be ready to intervene if something goes wrong.

## Concepts

### RunPod A100 Setup

The 7B model with GRPO (K=8) requires approximately 35–45GB VRAM. Unsloth's memory optimization reduces this to ~20–30GB with 4-bit quantization. An A100 80GB gives you comfortable headroom. An A100 40GB works with aggressive memory optimization.

Why RunPod over Colab for this run:
- Colab Pro sessions time out after 12 hours maximum
- GRPO training at K=8 for 1000 steps takes 6–10 hours for a 7B model
- RunPod allows SSH access for script modifications without losing progress
- A100 is 3–4× faster than T4 (Colab Free) for generation-heavy workloads like GRPO

### Memory Budget for GRPO

For a 7B model (Qwen2.5-Coder-7B-Instruct base) with Unsloth:
- Base model (4-bit): ~5GB
- LoRA adapters (r=16): ~200MB
- Reference model (4-bit, frozen): ~5GB
- K=8 completions in memory: ~3–6GB (depends on max_completion_length)
- Optimizer state (AdamW with LoRA parameters only): ~400MB
- Total: ~15–20GB with 4-bit, ~30–40GB with bf16

Use `load_in_4bit=True` for both the training model and reference model on A100 40GB. Use bf16 for the training model on A100 80GB (better precision, faster convergence).

### GRPO Monitoring Metrics

Key W&B metrics to watch during training:

**`mean_reward`:** Average reward across all completions in the current batch. Should start at your diagnostic baseline (from Week 47 Task 2) and trend upward. If it is flat for 100+ steps, something is wrong.

**`reward_std`:** Standard deviation of rewards within groups. If this approaches 0, all completions in each group are getting the same reward — zero gradient. This is the primary signal for "training is stuck."

**`kl_divergence`:** KL from reference model. Should stay below 5–10 nats. If it climbs rapidly, increase β.

**`policy_loss`:** The PPO clipping loss. Should fluctuate but not diverge. If NaN, training is failing.

**`grad_norm`:** Should stay below 1.0. Spikes above 5.0 indicate instability — check gradient clipping.

**`advantages_mean`:** Should be near 0 (because of normalization). If consistently positive or negative, normalization may be broken.

### Intervention Checklist During Training

If `mean_reward` is not improving after 100 steps:
1. Check `reward_std` — if near 0, your prompt distribution is too easy or too hard
2. Sample 5 raw completions from the model — has output format degraded?
3. Check KL divergence — if > 10 nats, increase β

If loss goes to NaN:
1. Stop training immediately
2. Restart from last checkpoint with gradient clipping: `max_grad_norm=0.5`
3. Reduce learning rate by 10×

If model starts generating very long completions (300+ tokens with no SQL):
1. Add a length penalty in the reward function: `reward *= max(0.5, 1 - 0.001*len(completion))`
2. Reduce `max_completion_length`

### Checkpointing Strategy

GRPO training is expensive. Checkpoint every 50 steps:

```python
GRPOConfig(
    save_steps=50,
    save_total_limit=5,   # Keep only last 5 checkpoints to save disk
    output_dir="./grpo_checkpoints",
)
```

Push the best checkpoint to HF Hub immediately after training. If RunPod session expires, you need the model.

### Evaluating v3

Use the same eval pipeline from Week 45 (the 200-query held-out test set). Run all three models:
- v1 (SFT only)
- v2 (SFT + DPO)
- v3 (SFT + DPO + GRPO)

The acceptance criterion: v3 outperforms v2 by at least 5 percentage points on execution accuracy on the full test set, AND outperforms v2 on the complex query tier specifically.

If v3 outperforms v2 on easy but not complex queries: the GRPO prompt set needs more complex examples. Note this for Week 50 iteration.

### Realistic Expectations

For a 7B model with K=8, 1000 GRPO steps on SQL:
- Expected execution accuracy improvement over v2: 5–15pp
- Expected semantic accuracy improvement: 3–10pp
- Expected training time on A100 80GB: 5–8 hours
- Expected cost on RunPod: $8–15

If v3 does not improve over v2 at all: do not panic. Note the state of your reward diagnostics (from Week 47) and use Week 50 to iterate. The first GRPO run is a learning run as much as a production run.

### The v3 Model and Phase 6

v3-grpo is the model you will build upon in Phase 6. Phase 6 adds:
- Dataset scaling (50K examples)
- A full SFT → DPO → GRPO pipeline run from scratch with the larger dataset
- Quantization and deployment

Your v3 from this week is the proof-of-concept. Phase 6 is the production run.

## Connections

Builds on: Week 47 (reward function), Week 46 (GRPO algorithm and TRL), Week 45 (eval pipeline).

Week 50 (Iteration): if v3 does not meet the acceptance criterion, Week 50 is the fix week.

Week 52 (Gate): v3 vs. v1 comparison is one of the Phase 5 Gate criteria.

## Common Misconceptions

- "If mean_reward goes up, the model is improving." Not always — check semantic accuracy on held-out data. The reward function can be gamed even with anti-hack guards.
- "I should train until loss converges." GRPO does not have a clean convergence criterion. Stop at 1000 steps and evaluate, then decide whether to continue.
- "RunPod A100 is too expensive." At $2–4/hour and 5–8 hours of training, the total is $10–32. This is within the Phase 5 budget allocation.
- "GRPO will always beat DPO on complex queries." If the GRPO prompt set lacks complex examples, it will not. Data distribution matters.

## Time Allocation (6–8 hours + async)

- 1 hour: Set up RunPod instance, install dependencies, upload scripts
- 30 min: Verify one training step completes without error
- 5–8 hours async: Let training run while you work on other tasks. Check W&B every 30–60 minutes.
- 1 hour: After training completes, push model to HF Hub, run eval pipeline
- 30 min: Write eval report comparing v1, v2, v3
