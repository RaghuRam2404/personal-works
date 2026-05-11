# Week 27 Quiz Answers — Phase 3 Gate

## Q1 — Answer: B

**Answer:** B — N=200M params, D=4B tokens.

**Why:**
```
Hours = 30 / 1.50 = 20 hrs
C = 0.35 × 312e12 × 20 × 3600 ≈ 7.9e18 FLOPs
N_opt = 6.8e-2 × (7.9e18)^{0.5} ≈ 6.8e-2 × 2.81e9.5 ≈ 191M ≈ 200M
D_opt = 1.96 × (7.9e18)^{0.5} ≈ 5.5e9 ≈ 5.5B tokens (rounds to ~4B at this precision)
```

**Why others are wrong:**
- A (500M): requires ~20× more compute
- C (50M, 1B): undershoots; your $30 buys more than Chinchilla-optimal for 50M
- D (7B, 140B): 7B params alone would need ~21e18 FLOPs for Chinchilla, far exceeding your budget

---

## Q2 — Answer: C

**Answer:** C — "Too many parameters for this dataset" is NOT a likely cause.

**Why:** Perplexity is not meaningfully affected by having "too many" parameters relative to dataset size during evaluation on held-out data. Having more parameters than Chinchilla-optimal for your compute budget means you trained less efficiently, but the resulting perplexity on val data reflects training token count and data quality, not an abstract over-parameterization issue. The other three options (A: insufficient training, B: tokenizer mismatch, D: domain mismatch) are all real causes of high perplexity.

---

## Q3 — Answer: B

**Answer:** B — Divided by the number of tokens to prevent bias toward shorter options.

**Why:** See Week 23. Length normalization divides cumulative log-likelihood by option length to convert from absolute log-probability (which favors short options) to average per-token log-probability. This enables fair comparison across options of different lengths in HellaSwag and similar benchmarks.

---

## Q4 — Answer: C

**Answer:** C — Only the assistant (SQL) turn tokens.

**Why:** See Week 25. Computing loss on user/system tokens would teach the model to generate user questions and system prompts. You want the model to improve at SQL generation; loss should be computed only on the tokens it is learning to produce — the assistant's SQL response.

---

## Q5 — Answer: C

**Answer:** C — All 671B parameters must be loaded, but only 37B are used per step.

**Why:** MoE models have all expert parameters in GPU memory (or spread across GPU memory in multi-GPU setups) because the router must be able to select any expert for any token. However, during each forward pass, only 37B active-parameter worth of computation occurs (the selected experts process the token). The 671B total params require ~1.34TB in FP16 to store — requiring multi-GPU setups. The 37B active compute means the throughput is similar to a 37B dense model.

**Why others are wrong:**
- A: correct that ~1.34TB of VRAM is needed for storage, but this requires multi-GPU (e.g., 16 A100s × 80GB = 1.28TB)
- B: the 37B is active compute, not stored-only; all 671B must be in GPU memory
- D: MoE models run on GPU; CPU inference is too slow for practical use

---

## Q6 — Short Answer (5-step pipeline)

1. **Download and convert Tier 1:** Load Spider and BIRD datasets with `load_dataset()`; convert to ChatML format using `spider_to_chatml()` and `bird_to_chatml()`; apply `sql_quality_filter()` and `filter_for_postgres_compat()`; dedup with `MinHashLSH(threshold=0.7)`. → 2,000 examples.

2. **Write and verify Tier 2:** Hand-write 100 PostgreSQL/TimescaleDB examples with realistic schemas; validate with `sqlglot.parse(dialect="postgres")`; execute each query against Docker PostgreSQL 16 to verify runtime correctness. → 100 examples.

3. **Generate Tier 3 with Self-Instruct:** Use `self_instruct.py` to generate 5,000 instruction candidates from the 100 seed examples; generate SQL responses with GPT-3.5 or Ollama/Qwen2.5-Coder-7B; apply quality filter; cross-deduplicate against Tier 1+2. → 2,900 examples.

4. **Merge and split:** Combine all tiers; shuffle with `random.seed(42)`; split 80/20 into `train.jsonl` (4,000) and `val.jsonl` (1,000).

5. **Publish to HuggingFace Hub:** Use `DatasetDict.push_to_hub(private=True)` with a dataset card (README.md) documenting sources, statistics, and quality assurance steps.

---

## Q7 — Short Answer (PPL=35, 3 causes)

1. **Most likely — Insufficient training tokens (< 1B processed):** Chinchilla optimal for 50M is 1.1B tokens. If you only processed 500M tokens, the model is undertrained. Diagnostic: check `train/tokens_seen` in W&B — if it did not reach 1B tokens, this is the cause.

2. **Second likely — Training data quality mismatch:** Your training data may have come from a different FineWeb-Edu shard than your validation data, or may have had different filtering applied. Diagnostic: compute perplexity on a 1,000-document sample from your training data itself — if train PPL is 20 but val PPL is 35, there is a distribution gap.

3. **Third likely — Tokenizer mismatch:** If you saved a tokenizer with different special tokens or vocab ordering than the one used during training, the evaluation will assign probability to the wrong token IDs. Diagnostic: encode and decode "Hello world" with the evaluation tokenizer and training tokenizer — if the token IDs differ, you have a mismatch.

---

## Q8 — Short Answer (ZeRO Stage 2)

ZeRO Stage 2 shards both the optimizer states (momentum and variance for AdamW) and the gradients across GPUs, with each GPU storing only 1/N of each. Each GPU still holds a full copy of the model parameters in memory, just as in standard DDP. This contrasts with DDP (which shards nothing and stores full optimizer states, gradients, and parameters on every GPU) and ZeRO-3 (which also shards the parameters themselves). ZeRO-2's advantage over DDP is memory efficiency on optimizer and gradient storage (the largest memory consumers) without the per-step all-gather communication overhead that ZeRO-3 requires to reconstitute parameters before each forward pass.

---

## Q9 — Comprehensive Scenario Model Answer

**1. Is the training run acceptable?**
Yes, with caveats. Val loss 3.1 (PPL ≈ 22.2) is excellent for a 50M model trained on 3B tokens — it exceeds the Phase 3 minimum bar (val loss < 4.0). However, training on 1 epoch of a 3B-token dataset means the model has seen each document only once, which is typically fine for language modeling (no memorization risk). The training run is acceptable; ideally 2B tokens on a more diverse dataset would be preferred, but 3B on a single epoch is defensible.

**2. Is the 3,800-example dataset acceptable?**
CONDITIONAL. The Phase 3 goal is 5,000 examples. 3,800 is close but short, and 65 hand-written examples (vs. the target of 100) means the highest-quality tier is underbuilt. For Phase 3 gate purposes, a CONDITIONAL PASS is appropriate — the dataset is functional for early SFT experiments (Week 29), but the junior should complete the remaining 1,200 examples (including 35 more hand-written ones) before starting full fine-tuning in Week 31+.

**3. Evaluating the "better than GPT-2 per parameter" claim:**
This claim is misleading. The comparison is not apples-to-apples because: (a) GPT-2 was trained on only 40B tokens (WebText), far less than 3B per-param; the junior's model was trained on 3B tokens, which is 60× more tokens per parameter (3B/50M = 60) vs. GPT-2's 40B/117M = 342 tokens/param — actually far fewer tokens per parameter. (b) The similar benchmark scores (32% vs 31.6% HellaSwag) are within noise — 1,000 evaluation examples at 25% random baseline have substantial variance. (c) Correct claim: "My 50M model trained on 3B tokens achieves similar HellaSwag performance to GPT-2-small, suggesting my data pipeline is reasonably effective." The "per parameter" framing is irrelevant.

**4. Recommendation on fine-tuning the 50M model:**
Do not fine-tune this 50M model on the SQL dataset. The Phase 3 pretraining experience was educational — training a 50M model from scratch teaches the pipeline, debugging, and evaluation skills. But the fine-tuning target for Phase 4+ is Qwen2.5-Coder-7B, which starts from 5.5T tokens of code-focused pretraining. Fine-tuning the 50M model would produce an inferior SQL assistant (50M parameters cannot learn complex PostgreSQL semantics at the required depth), and the work would be thrown away when moving to the 7B model. Use the 50M model only for: (a) testing your fine-tuning pipeline before running it on the expensive 7B model; (b) verifying your dataset format is correct.

**5. Should MMLU 27% be concerning?**
No. MMLU at 25% random baseline for a 4-choice task. A 50M parameter model achieving 27% on MMLU is exactly what theory predicts — the model lacks the parameter capacity to store the factual knowledge tested by MMLU. GPT-3 (175B params) scores only 43% on MMLU. The 27% result is not a sign of a failed training run; it is a confirmation that your model is small and trained on general web text without domain-specific factual learning. MMLU is simply the wrong benchmark for evaluating a 50M model's quality — perplexity and HellaSwag are more appropriate.
