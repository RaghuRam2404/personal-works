# Week 40 — Phase 4 Gate: Consolidation and Readiness Check

## Learning Objectives

By the end of this week, you will be able to:

- Verify that your Phase 4 deliverables are complete and meet the quality bar for Phase 5 entry
- Explain the end-to-end fine-tuning pipeline you built — from raw data to evaluated model — to a technical peer
- Identify the specific weak points in your current `postgres-sqlcoder-7b-v1` model and name the Phase 5 techniques that address each
- Push your best adapter to HuggingFace Hub and write a model card that accurately describes training, evaluation, and limitations
- Reflect on what you learned in Weeks 28–39 and identify the 3 concepts that are likely to trip you up in Phase 5

---

## What This Week Is

Week 40 is a gate, not a new-content week. Phase 5 introduces GRPO, reward modeling, and inference-time search — techniques that require a solid fine-tuning foundation to use correctly. If your foundation has gaps (no working eval harness, no 7B fine-tune, no domain dataset), Phase 5 will be frustrating and unproductive.

This week you complete any outstanding deliverables, run the gate checklist, and publish your model. The "assignment" is a self-audit followed by a public artifact.

There is no new theory this week. Use the time to consolidate: re-read your own code from Weeks 33–39, clean it up, and make sure you could reproduce any result from memory.

---

## Phase 4 Gate Checklist

Go through each item. If any are incomplete, fix them before moving to Phase 5.

### Minimum Requirements (all must pass)

- [ ] **At least 4 fine-tuning runs completed:** Full SFT on a tiny model (Week 29), LoRA from scratch (Week 30), LoRA via peft (Week 31), QLoRA on a 7B model (Week 33 or 38). You do not need all to be domain-tuned — the 4-run requirement is about technique variety.

- [ ] **Working 7B fine-tune on Colab Pro:** Your Week 33 or Week 38 QLoRA run on `Qwen/Qwen2.5-Coder-7B` completed without OOM, logged to W&B, and produced a checkpoint. If you only ran the 1.5B model, you need to re-run with the 7B before entering Phase 5.

- [ ] **Working execution-based eval harness:** `eval_harness.py` from Week 39 runs end-to-end on 100 examples, produces `exec_success %` and `exec_correct %`, and is committed to your GitHub repo.

- [ ] **v1 model on HuggingFace Hub:** `<your-handle>/postgres-sqlcoder-7b-v1` is public (or unlisted), has a model card, and the adapter loads correctly via `PeftModel.from_pretrained`.

- [ ] **15K-example domain dataset committed:** `train_15k.jsonl`, `val_500.jsonl`, and `held_out_test.json` are in your repo or on HuggingFace Datasets. The held-out test set was never used in training.

### Quality Bar (aim for these)

- Execution correctness on `held_out_test.json`: 60% or higher for your Week 38 model
- W&B logs for at least 3 runs publicly viewable or exported to your repo as screenshots
- At least one rank sweep result documented (Week 31 or 35)

---

## Concepts to Review Before Phase 5

Phase 5 builds directly on these — if they feel fuzzy, re-read the relevant week's Curriculum.md now.

### LoRA Rank and Capacity

You chose rank 16 throughout most of Phase 4. Recall why: lower rank = fewer trainable parameters = less overfitting risk, but also less capacity to learn complex SQL patterns. In Phase 5, GRPO training updates the model based on reward signals — the LoRA adapter needs enough capacity to absorb those reward-shaped gradients. If you are uncertain about the rank-vs-capacity tradeoff, re-read Week 30 and Week 31.

### QLoRA's Frozen Quantized Base

The base model weights in QLoRA are quantized to NF4 and frozen. Only the LoRA adapter weights are trained in BF16. In Phase 5, you may load the same NF4 base and attach a new adapter for GRPO training. Understanding why the base stays frozen (memory efficiency) and what its gradient does not flow through (the base itself) is critical for Phase 5 debugging.

### The Eval Harness as a Reward Signal

Your Week 39 harness returns a binary `exec_correct` flag per example. In Phase 5's GRPO training, this flag becomes the reward: the model generates multiple SQL candidates, the harness executes each, and reward is 1.0 if exec_correct and 0.0 otherwise. The cleaner and faster your harness, the more GRPO training iterations you can run per hour. Re-read Week 39's Curriculum.md on harness architecture and safety checks.

### Hyperparameter Sensitivity

Phase 5 training is more sensitive to learning rate than SFT. GRPO uses small LRs (1e-6 to 5e-6) because reward-shaped updates are noisy. Your Week 35 intuition about LR and loss divergence applies directly.

---

## What Phase 5 Will Introduce (Preview, Not Detail)

Phase 5 (Weeks 41–52) covers:

- **GRPO:** Group Relative Policy Optimization — fine-tune with verifiable rewards. Your exec harness is the verifier.
- **Reward modeling:** What makes a good reward function for SQL (execution correctness + efficiency bonus + formatting).
- **DPO:** Direct Preference Optimization — teach the model to prefer one SQL over another without a reward model.
- **Inference-time search:** Generate N SQL candidates, execute all, pick the best — a strong baseline that improves results without any further training.

All of Phase 5 assumes you have a working fine-tuned model and a working eval harness. Both are now true for you.

---

## Connections

**Built on:** Every week from 28–39. This week synthesizes rather than extends.

**Required for Phase 5:** A clean HuggingFace adapter, a working eval harness, and conceptual clarity on QLoRA and LoRA rank.

---

## Common Misconceptions / Pitfalls

- "Phase 4 is done when I finish the assignments." Done means the gate checklist passes. If your 7B fine-tune OOM'd in Week 33 and you never retried, that gap will cost you in Phase 5.
- "My model card doesn't matter." A model card is your technical contract: training data, prompt format, known limitations. If you load the adapter in Phase 5 without knowing its expected prompt format, inference will produce garbage.
- "I should start Phase 5 now and come back to the gate." Phase 5's first assignment uses the eval harness as a GRPO reward function. If the harness is broken or slow, Week 41 stalls immediately.

---

## Time Allocation (6–8 hrs)

| Activity | Time |
|---|---|
| Run the gate checklist; fix any gaps | 2h |
| Write/finalize the HuggingFace model card | 1h |
| Push model, dataset, and eval harness to Hub/GitHub | 1h |
| Re-read Weeks 30, 33, and 39 Curriculum.md for consolidation | 1.5h |
| Write a personal retrospective: 3 things you learned, 3 things that surprised you, 3 gaps you still feel | 1h |
| Preview Phase 5 Week 41 Curriculum.md (read-ahead only) | 30m |
