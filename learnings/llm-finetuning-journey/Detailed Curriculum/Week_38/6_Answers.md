# Week 38 Quiz Answers

## Q1 — Answer: C

**Answer:** C. No — 0.02 improvement is within noise; model has converged.

**Why:** A 0.02 loss improvement between epochs on a validation set of 500 examples is within typical measurement noise (±0.03–0.05 depending on batch size and dataset variability). The model has reached the generalization limit of this dataset+architecture combination. Running a third epoch would risk: (1) slight overfitting, (2) unnecessary compute cost, (3) marginal loss gain that doesn't translate to meaningful quality improvement on held-out evaluation. Save the checkpoint from epoch 2 and proceed to evaluation.

---

## Q2 — Answer: B

**Answer:** B. Misleading — Phase 4 goal is beating the base model, not GPT-4o.

**Why:** The Phase 4 acceptance criterion is: "Your fine-tuned 7B beats the base Qwen2.5-Coder-7B." The base model scores 5–15% exact match; your model scores 55% — a 4–10x improvement. GPT-4o beating is the Phase 6 goal (with 100K+ examples and GRPO on verifiable rewards). Comparing to GPT-4o-mini at this stage missets expectations. The correct framing: at 55%, you have demonstrated that SFT on 15K examples dramatically improves SQL generation. The path to beating GPT-4o runs through execution-based RL (Phase 5–6), not more SFT epochs.

---

## Q3 — Answer: B

**Answer:** B. Malformed batch causing large gradient; add `max_grad_norm=1.0`.

**Why:** A single training example that is anomalously long, contains unusual tokens, or has a pathological loss value can produce a very large gradient for that batch. This causes a temporary loss spike as the optimizer takes an oversized step, followed by recovery in subsequent steps as normal batches drive the loss back down. Gradient clipping (`max_grad_norm=1.0`) caps the gradient norm to 1.0 before each parameter update, preventing any single batch from causing catastrophic weight changes. This should be included in every production training script.

---

## Q4 — Answer: B and D are both partially correct; B is the cleaner answer.

**Answer:** B. Explain adapter loading pattern.

**Why:** The standard PEFT deployment pattern: (1) user downloads the base model (`Qwen/Qwen2.5-Coder-7B`) from HuggingFace separately, (2) loads the adapter with `PeftModel.from_pretrained(base_model, "your-handle/postgres-sqlcoder-7b-v1")`. Option D is also correct but less precise — the user downloads the base model from HuggingFace automatically when they call `from_pretrained("Qwen/Qwen2.5-Coder-7B")`. This two-step pattern is standard for PEFT adapters and should be documented in the model card.

---

## Q5 — Answer: B

**Answer:** B. Add examples with multiple plausible column names requiring correct selection.

**Why:** "Wrong column referenced" errors indicate the model is choosing the wrong column from the schema — perhaps selecting `total_price` when the question asks for `unit_price`, or using `customer_id` when `user_id` is the correct column. To fix this, add training examples where: the schema has multiple similarly-named columns (price/total_price/unit_price), and the question specifically discriminates between them. Examples that require careful schema reading improve the model's attention to column name specifics.

---

## Q6 — Short Answer

If exact match is 58% and execution correctness is typically 10–20 percentage points higher, estimated execution correctness is 68–78%. The lower bound (68%) assumes minimal false positives from exact match; the upper bound (78%) assumes significant semantic equivalence that exact match misses (e.g., column aliases, quote styles).

For the Phase 4 gate: the criterion is "your fine-tuned 7B beats the base Qwen2.5-Coder-7B on your held-out PostgreSQL test set." With base model execution correctness at approximately 15–25% and your model at estimated 68–78%, you have decisively met this criterion. The exact-match metric (58%) is sufficient evidence even without running the full execution-based eval — though Week 39 will provide the execution-based number as your definitive metric.

---

## Q7 — Short Answer

Your model's 25% exact match on TimescaleDB examples (2/8) vs. 58% overall tells you that your Week 37 dataset had insufficient TimescaleDB-specific examples to teach the model these patterns. 30–100 TimescaleDB examples out of 15K total is less than 1% of training data — far too little to learn `time_bucket`, hypertable queries, and continuous aggregates reliably.

For v2 fine-tune changes:
1. Increase TimescaleDB examples from 30–100 to at least 500–1,000 (3–7% of dataset)
2. Focus on covering all key TimescaleDB functions: `time_bucket`, `time_bucket_gapfill`, `last`, `first`, continuous aggregates, hypertable compression
3. Upsample existing TimescaleDB examples 10–20× in the training loop to weight them higher
4. Add evaluation examples specifically for TimescaleDB and track this as a separate metric

---

## Q8 — Short Answer

Merging and pushing the full model (14GB) is the right choice when:
- You want zero-friction inference for end users (no two-step load)
- You are ready to deploy and don't plan further adapter updates
- Storage cost is acceptable ($0.10/GB on HuggingFace for large files)
- The model is a final release, not an experimental checkpoint

Pushing only the adapter (50–100MB) is the right choice when:
- You are still iterating (v1, v2, v3) — each iteration only saves the small adapter
- Users likely already have the base model loaded (in a multi-adapter deployment)
- Storage and bandwidth costs matter
- You want to demonstrate the PEFT workflow for educational purposes (this course)

For Week 38: push the adapter only. You will continue iterating with execution-based RL in Phase 5. A merged full model push makes more sense for the final Phase 6 release.

---

## Q9 — Scenario Answer

Eval loss rising from step 100 to 200 is a warning sign, but may not be definitive yet. The right action at step 200:

**Immediate:** Do NOT stop training — you are only 1/3 through the planned run. Eval loss can fluctuate early. Continue to step 300 and re-evaluate.

**Diagnose:** Check if the eval loss at step 200 (2.5) is above the starting value (2.4). It is slightly higher — concerning but not catastrophic.

**If eval loss at step 300 is 2.6–2.7 (continuing to rise):** Stop training. The model is overfitting on the 15K dataset, which should not happen this early — check: (1) Is the eval set accidentally contaminated with training data? (2) Is the LR too high (loss should not diverge for LR=2e-4 with standard dataset)? (3) Is packing causing cross-example attention issues?

**If eval loss at step 300 stabilizes at 2.4–2.5:** Continue to completion. Early eval fluctuation is common.

**Most likely cause of early eval loss rise:** The val_500 set has slightly different schema distribution than the training set — eval loss measures out-of-distribution generalization which starts high and improves as the model sees more schema diversity in training.

**Action if stopping:** Save the best checkpoint (step 100, eval loss 2.3), which `load_best_model_at_end=True` will handle automatically. Proceed with that model to Week 39.
