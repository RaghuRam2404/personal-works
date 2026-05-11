# Week 21 Quiz Answers

## Q1 — Answer: B

**Answer:** B — 11 hours.

**Why:** 2B tokens / 50,000 tokens/sec = 40,000 seconds = 11.1 hours. This is the standard estimate for a 56M model on an A100 at ~40% MFU. Actual time depends on your exact throughput and Python overhead.

**Why others are wrong:**
- A (4h): would require ~140K tokens/sec — unrealistic for a single A100 on a 56M model
- C (22h): would imply 25K tokens/sec — possible only if running without mixed precision or with very large batch sizes
- D (40h): training is much faster than this; check if you forgot to use mixed precision

---

## Q2 — Answer: B

**Answer:** B — Log the spike and continue.

**Why:** A transient loss spike that recovers within 200 steps is almost certainly caused by a single bad batch — an unusually long document, a corrupted token, or a rare character sequence that produces extreme logits. The model's gradient dynamics are self-correcting for small spikes. Stopping and restarting costs you the progress made after the spike and introduces unnecessary disruption. The correct action is to note the step number, magnitude, and recovery time.

**Why others are wrong:**
- A: stopping is premature; you lose 5,000 steps of progress unnecessarily
- C: reducing batch size does not address bad batch issues and slows training
- D: increasing clip to 5.0 would actually make spikes worse by allowing larger gradient magnitudes

---

## Q3 — Answer: B

**Answer:** B — Has a wider dynamic range (8-bit exponent vs. 5-bit).

**Why:** BF16 uses 1 sign + 8 exponent + 7 mantissa bits vs. FP16's 1+5+10. The 8-bit exponent gives the same dynamic range as FP32 (both have 8 exponent bits). This eliminates the overflow/underflow issues common in FP16, making BF16 far more stable for training without a GradScaler. The 7-bit mantissa (vs. 10-bit in FP16) means slightly lower precision per operation, but for deep learning training this rarely matters.

**Why others are wrong:**
- A: bf16 has fewer mantissa bits (7 vs 10) — lower precision
- C: GradScaler is needed for fp16, not bf16
- D: A100 has tensor cores for both; bf16 training speed is comparable to fp16

---

## Q4 — Answer: B

**Answer:** B — The learning rate is too high relative to the current loss landscape.

**Why:** Gradient clipping fires when the gradient norm exceeds the clip threshold (1.0). If this happens every step, it means the optimizer is consistently trying to take steps that are too large — a sign that the learning rate is in a regime where the loss surface is steep. After the warmup phase, the LR should have settled; if clipping continues through 5K+ steps, reduce max_lr by 30%.

**Why others are wrong:**
- A: clipping every step is not normal beyond the first few hundred warmup steps
- C: corrupted tokens would appear as irregular spikes, not consistent clipping
- D: bf16 overflow would produce NaN gradients, not large finite gradients

---

## Q5 — Answer: B

**Answer:** B — The model will train on some data a third time, adding a fractional extra epoch.

**Why:** At step 15,000, you have seen ~15,000 × 65,536 ≈ 983M tokens. With an 800M-token train.bin, you are already ~1.2 epochs in. Resetting the DataLoader means you restart from token 0 in the dataset. Some tokens from the beginning of the dataset have now been seen 3 times instead of 2. This is mild over-training but not catastrophic — the model has robust gradient history from the optimizer states.

**Why others are wrong:**
- A: order does not matter for language model training (unlike online algorithms)
- C: catastrophic forgetting requires learning a completely different task; seeing the same data again does not erase knowledge
- D: optimizer states are saved and loaded correctly; they are decoupled from data order

---

## Q6 — Short Answer

Perplexity = exp(val_loss) = exp(3.2) ≈ 24.5.

Intuitively, perplexity of 24.5 means that on average, your model behaves as if it were choosing uniformly among about 24 equally likely next tokens at each position. A perfect model would have perplexity 1 (certainty). A completely random model would have perplexity 32,000 (vocab size). GPT-2 small (124M params) achieves perplexity ~18–20 on WebText; your 56M model at 25 is slightly worse, which is expected given it is smaller. The number is not the goal — experiencing the full training pipeline is.

---

## Q7 — Short Answer

**Hypothesis 1: Dataset is too small and the model has memorized it.**
Diagnostic: Check `tokens_seen / dataset_size_tokens`. If > 3 (more than 3 epochs), overfitting is likely. Action: check `train_loss - val_loss`; if this gap is growing, overfitting is confirmed.

**Hypothesis 2: Learning rate has decayed too aggressively.**
Diagnostic: Check `train/lr` in W&B at step 10,000. If it is below 3e-5 (the min_lr), the cosine schedule has bottomed out before enough training. Check your `max_steps` parameter — you may have set it too low, causing premature decay.

**Hypothesis 3: Gradient accumulation is broken, producing lower effective LR.**
Diagnostic: Compare expected effective batch size (batch × accum × block) with tokens_seen increment per step. If `tokens_per_step` is half the expected value, your accumulation loop has a bug and you are effectively halving the gradient magnitude each step.

---

## Q8 — Short Answer

Gradient clipping (`max_norm=1.0`) limits the L2 norm of the gradient vector to 1.0 before the optimizer step. If the gradient norm is 5.0, it scales all gradients by 0.2. This prevents the optimizer from taking a huge step that overshoots the loss minimum and destabilizes training.

Setting `grad_clip=10.0` means only gradients larger than 10.0 in norm will be clipped — this is essentially no clipping for most training scenarios. During early training when gradients can be large, unclipped gradients cause parameter updates that push weights far from their current values, resulting in NaN or divergent loss. The standard `max_norm=1.0` is not "preventing gradients from flowing" — it is rescaling them to a manageable magnitude while preserving direction.

---

## Q9 — Scenario Model Answer

**1. Tokens wasted:** Steps 6,000 to 8,000 = 2,000 steps × 65,536 tokens/step ≈ 131M tokens re-computed. At 50K tokens/sec, this costs ~44 minutes of compute. Acceptable, but motivates more frequent checkpointing.

**2. Risk of triple-seeing tokens:** Steps 0–8,000 = 524M tokens processed (before disconnect). With 800M token dataset, you are at step 8,000 on epoch 1 (about 65% through the dataset). After resume from step 6,000, you re-process 131M tokens from the dataset starting at position 393M. These tokens were already processed in the original run. Combined with the reset-to-zero DataLoader behavior after a second resume, the first 393M tokens of the dataset will have been seen approximately 1.17× on average — not quite triple, but some tokens near the start may be seen 2+ times if the DataLoader is reset. This is mild and acceptable at this training scale.

**3. Problem with missing optimizer states:** Without Adam's momentum (m_t) and variance (v_t) states, the optimizer restarts with zero estimates of gradient history. The first few hundred steps after resume will have artificially large effective learning rates (since v_t starts at 0, and Adam's update is lr × m_t / (sqrt(v_t) + eps) — with v_t=0 the denominator is just eps, making updates very large). This causes a temporary loss spike and potentially divergence. The solution: always save optimizer states with model weights.

**4. Simple two-checkpoint strategy:**
```
checkpoints/
  latest.pt     # overwritten every N steps
  backup.pt     # overwritten every 2N steps (staggered from latest)
```
Before overwriting `latest.pt`, first move it to `backup.pt`. This ensures you always have two recent checkpoints: if `latest.pt` is corrupt, use `backup.pt`. Both must include model weights AND optimizer states.
