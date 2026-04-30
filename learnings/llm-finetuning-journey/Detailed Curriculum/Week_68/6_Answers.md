# Week 68 Answers

## Q1

**Answer: C**

**Why correct:** Gradient clipping and weight decay are optimizer hyperparameters that belong in the training pipeline section's hyperparameter table. These values directly affect training stability and convergence — a different gradient clipping threshold can change whether training diverges or converges. Reproducibility reviewers specifically look for: optimizer name (AdamW), beta values (0.9, 0.999), epsilon (1e-8), weight decay (0.01 or 0.1), and gradient clipping threshold (usually 1.0). Omitting them makes the training pipeline section incomplete by ML publication standards.

**Why others are wrong:**
- A: Gradient clipping is an optimizer setting, not a data preprocessing concern.
- B: These values affect training, not evaluation/inference time behavior.
- D: Optimizer details are required in any reproducible ML paper; they are not optional.

---

## Q2

**Answer: B**

**Why correct:** Wall-clock time = number of steps × time per step. Time per step is determined by tokens processed per step divided by tokens per second (throughput). With step count (2,400) and throughput (tokens/second, or equivalently, seconds/step from W&B's step timing logs), the reviewer can compute: total time = 2,400 steps × seconds/step. This is why W&B's "time/step" metric is worth logging — it converts steps to wall-clock time for reproducibility.

**Why others are wrong:**
- A: Final validation loss tells you quality, not speed.
- C: LoRA rank affects parameter count, which can affect speed marginally, but alone does not enable time computation.
- D: The LR schedule affects convergence, not per-step compute time.

---

## Q3

**Answer: B**

**Why correct:** The DPO paper (Rafailov et al. 2023) defines β as the coefficient scaling the KL divergence between the current policy and the reference policy (the frozen SFT checkpoint). Small β (0.01–0.1) allows large policy updates; large β (1.0+) keeps the policy close to the reference. Your paper should include this clarification: "β controls the KL penalty between the trained and reference policy; β=0.1 allows moderate deviation from the SFT checkpoint." This makes the value interpretable to researchers using different DPO implementations.

**Why others are wrong:**
- A: Standard DPO uses a reference model; reference-free DPO is a different variant and should be named explicitly.
- C: β is not a learning rate multiplier — it has a specific probabilistic interpretation.
- D: This is advice, not a clarification about what β means.

---

## Q4

**Answer: B**

**Why correct:** The claim "four-stage pipeline outperforms SFT-only" is an ablation claim. An ablation claim requires a controlled experiment where only the claimed component differs — in this case, training with SFT alone (same data, same base model) vs training with all four stages. Showing that DPO loss decreases (C) or GRPO reward increases (D) only proves the optimization is working, not that the final model is better than SFT-only. Comparing to GPT-4o (A) tests your model vs a different system, not the contribution of additional training stages.

**Why others are wrong:**
- A: GPT-4o comparison tests overall quality but not the specific pipeline contribution.
- C: DPO loss decrease is expected during training; it does not prove accuracy improvement over the SFT checkpoint.
- D: Same issue as C for GRPO.

---

## Q5

**Model answer:** LoRA's effective weight update is W = W_0 + (α/r) × BA, where A and B are the low-rank matrices, r is the rank, and α is the scaling factor. The α/r ratio controls the magnitude of the adapter's contribution relative to the frozen base weights. When α = r, the scaling factor is 1.0, meaning the adapter contribution is neither amplified nor suppressed. The convention α = 2r (e.g., rank=64, alpha=128) doubles the adapter's effective learning rate, which empirically helps adapters at rank ≥ 32 where the gradient signal per parameter is smaller. The QLoRA paper established this convention because it allows the optimizer to use the same base learning rate across different rank values — you do not need to re-tune the LR when changing rank.

---

## Q6

**Model answer:** "Our training cost is low because we train only low-rank adapter matrices (LoRA) rather than full model weights, and we start from an already-capable base model. Full pre-training a 7B model from scratch on trillions of tokens requires thousands of GPU-hours; our fine-tuning modifies less than 1% of parameters for 12.4 GPU-hours total. This demonstrates that domain adaptation via post-training is dramatically more compute-efficient than pre-training, and is accessible to individual researchers with standard cloud GPU budgets."

---

## Q7

**Model answer:** Increasing K from 8 to 16 doubles the number of SQL completions sampled per prompt in each GRPO step. The benefit is that the reward signal becomes more reliable — with K=16, you have a better estimate of the average reward across the completion distribution, which reduces variance in the policy gradient estimate and can improve training stability. The cost is that compute per step doubles: you run 16 forward passes instead of 8 per prompt, and the KV cache memory requirement scales with K. In practice, doubling K from 8 to 16 typically yields 0.5–1.5 pp accuracy improvement at 2x training time — whether this tradeoff is worth it depends on your compute budget. K=8 is the established default in GRPO papers on similar tasks.

---

## Q8 — Deep Scenario

**Model answer:**

4.4 Direct Preference Optimization

We apply DPO (Rafailov et al. 2023) to the SFT checkpoint using 5,000 preference pairs, where chosen responses are SQL queries that executed correctly and rejected responses produced wrong results on the test database. We use β=0.1, which controls the KL penalty between the trained and reference policy; preliminary experiments at β ∈ {0.05, 0.1, 0.5} showed β=0.1 provided the best balance between staying close to the SFT prior and learning from preference signal.

After 800 training steps, the model achieves a log-ratio reward margin of 0.23, meaning that on average, the model assigns 0.23 higher log-probability to chosen over rejected responses. Concretely, this translates to a 2.8 pp accuracy improvement on our 200-example TimescaleDB benchmark compared to the SFT-only checkpoint (SFT: 80.3%, DPO: 83.1%). This improvement is concentrated on time-series queries, where DPO successfully shifted the model away from standard SQL patterns toward TimescaleDB-specific aggregations that the rejected completions incorrectly approximated.
