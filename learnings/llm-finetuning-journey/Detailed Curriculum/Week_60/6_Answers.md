# Week 60 Answers

## Q1 — Answer: B

**Why:** GRPO's advantage estimate for each candidate is computed relative to the group mean reward. If all 8 candidates have the same reward (2.1), the group mean is 2.1, and every candidate's advantage is 2.1 - 2.1 = 0. Zero advantage → zero policy gradient → no parameter update. This is why GRPO prompts should be chosen from examples where the model sometimes succeeds and sometimes fails — reward variance within the group is the signal.

---

## Q2 — Answer: B

**Why:** GRPO requires K=8 full inference passes (generation) + backward passes per prompt. At H100 throughput, generating 8 candidates per prompt takes approximately 8× longer than SFT's single-pass training. 25,000 prompts × 8 candidates = 200,000 inference calls, which would take ~20+ hours on H100 at typical speeds — far beyond the budget. The practical ceiling for a single GRPO run is 1,000–3,000 prompts.

---

## Q3 — Answer: B

**Why:** KL divergence > 10 bits indicates the policy has drifted far from the reference model — the DPO-v3 distribution and the GRPO policy are now substantially different models. The KL constraint (kl_coef) is not pulling back hard enough. The fix: increase kl_coef (more KL penalty per step) and resume from an earlier checkpoint where KL was still bounded. Do not continue from step 200 — the current policy may have lost valuable DPO alignment properties.

---

## Q4 — Model Answer

With binary reward (0 or 1) and group size K=4, it is entirely possible that all 4 candidates either all fail (reward 0) or all succeed (reward 1). In either case, all advantages are 0 and there is no gradient update — a completely wasted step.

Partial credit (0.2 for "executes but wrong rows") introduces gradient information even when no candidate achieves full credit. If candidates score [0.0, -0.5, 0.2, 0.2], there is variance within the group: the two partially-correct candidates get positive advantage relative to the ones that failed. The model learns to prefer "generate SQL that at least executes with the right schema structure" over "generate syntactically broken SQL." This gradient is weaker than a full-credit signal but is far better than zero.

---

## Q5 — Model Answer

Yes, this is a success — by the metrics that matter for your use case.

The comparison "7B at 78% vs. GPT-4o at 83%" understates your achievement. GPT-4o is estimated at 200B+ parameters, trained on trillions of tokens, and costs ~100× more per inference call. Your 7B model at 78% on the same benchmark represents roughly 94% of GPT-4o's performance at 1–2% of the inference cost. On your most critical subset (TimescaleDB-specific queries), your model at 70% likely outperforms GPT-4o on the same questions — a 200B generalist model has seen far fewer TimescaleDB-specific examples than your purpose-built 7B model.

The meaningful success criteria for this project: your 7B model outperforms a general-purpose 7B model (base Qwen2.5-Coder-7B at perhaps 45–50%) by a very large margin, outperforms GPT-4-class models on TimescaleDB-specific tasks, and runs locally on your Mac in quantized form. All three criteria are achieved.

---

## Q6 — Model Answer

The 14.2GB merged model is stored in bf16 (bfloat16, 2 bytes per parameter). The 28.4GB model is stored in float32 (4 bytes per parameter). Both are correct representations of the same model. For inference, bf16 is preferred — it cuts memory in half with negligible precision loss for inference. The 28.4GB float32 version would be used only if very high-precision inference were required (e.g., for computing gradients in subsequent training, though you'd use bf16 training even then).

Qwen2.5-Coder-7B has approximately 7.6B parameters. At bf16: 7.6B × 2 bytes ≈ 15.2GB (accounting for architecture overhead, 14.2GB is plausible for the base transformer layers only without embeddings in separate shards, or with different precision for specific layers).

---

## Q7 — Technical Report Discussion

**Paragraph 1 — Training pipeline contributions:**
The progression from Phase 5 GRPO (59%) to the final model (76%) reflects the compounding effect of the three-phase Phase 6 training pipeline. Dataset v3 (Weeks 53–56) contributed the largest single gain: the transition from Phase5-GRPO to SFT-v3 alone yielded +12 percentage points on the custom benchmark, primarily attributable to the increase in dataset quality and diversity (50K filtered, multi-turn examples vs. the Phase 5 dataset of approximately 15K examples). The continued pretraining step (Week 57) is estimated to have contributed approximately 3 of these 12 points, based on the held-out domain perplexity improvement measured before SFT. DPO (Week 59) and the final GRPO step (Week 60) together contributed +5 percentage points, consistent with prior work showing DPO and GRPO add incremental refinement over a strong SFT baseline rather than large absolute gains.

**Paragraph 2 — TimescaleDB subset gap:**
The TimescaleDB subset consistently underperforms the overall custom benchmark by 6–8 percentage points across all model versions. This gap is attributable to two factors. First, despite targeted efforts to include ≥3,000 TimescaleDB-specific examples in v3, TimescaleDB content remains a minority of the training data (~12%), and the model's prior (from Qwen2.5-Coder-7B pretraining) strongly reflects standard PostgreSQL patterns. Second, TimescaleDB hyperfunctions — particularly `time_bucket_gapfill`, `locf`, and `interpolate` — have subtle argument order requirements and interaction with GROUP BY that are not well-covered even in the teacher model's generated data. Future iterations should increase TimescaleDB representation to ≥25% of training data.

**Paragraph 3 — Remaining gap to GPT-4o:**
The 7-point gap between GRPO-final (76%) and GPT-4o (83%) on the custom benchmark decomposes into two components: model capacity and training distribution. Approximately 3–5 points are attributable to model capacity — a 7B model with finite context and attention heads cannot represent the same breadth of SQL reasoning as an estimated 200B-parameter model. The remaining 2–4 points reflect training distribution: GPT-4o's RLHF training covers a broader range of SQL patterns, including novel schema structures, than our 50K training set. Notably, on the TimescaleDB-specific subset — where our domain training is most concentrated — the gap narrows to 13% vs. GPT-4o, and we estimate our model may match or exceed GPT-4o on queries involving TimescaleDB hyperfunctions specifically, though a controlled comparison of this specific subset was not conducted. We leave a systematic GPT-4o comparison on the TimescaleDB subset for future work.
