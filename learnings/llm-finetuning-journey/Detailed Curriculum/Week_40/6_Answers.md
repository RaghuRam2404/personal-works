# Week 40 Quiz Answers

## Q1

**Answer: B**

**Why:** Rank 16, alpha 32 (ratio 2:1), all 7 projection modules, and LR 2e-4 is the established robust default for LoRA on 7B models. Targeting all 7 modules (attention + MLP) captures both the model's knowledge retrieval (attention) and its generation patterns (MLP). The 2:1 alpha-to-rank ratio ensures the effective learning rate of the LoRA update is appropriately scaled. On 24GB VRAM, rank 16 on 7 modules adds approximately 20–40M trainable parameters — well within compute budget.

**Why others are wrong:**
- A: Rank 4 with only q/v gives insufficient capacity for a complex SQL domain with 10K examples. This under-parameterizes the adaptation.
- C: Rank 64 is aggressive for 10K examples and risks overfitting. LR 5e-4 is too high for LoRA on a 7B model and will likely cause divergence.
- D: Alpha equal to rank (both 16) gives a 1:1 ratio, which effectively under-scales the LoRA update. LR 1e-5 is appropriate for full fine-tuning, not LoRA.

---

## Q2

**Answer: C**

**Why:** Loss dropping from 1.8 to 1.3 in 200 steps then plateauing at 1.2 is the textbook pattern of a model that has learned the in-distribution patterns and hit the information ceiling of the training set. With 10K examples and rank 16, the model has absorbed what it can. The plateau is not due to optimizer or quantization failure — those typically manifest as spikes or oscillations, not smooth plateaus. The correct intervention is more or better training data, not hyperparameter changes.

**Why others are wrong:**
- A: NF4 quantization error is small and does not worsen over training steps — it is a fixed approximation error on base weights, not an evolving degradation.
- B: LR-induced overshooting shows as loss oscillation or a spike followed by chaotic variation, not a smooth plateau at 1.2.
- D: paged_adamw_8bit is well-tested and does not degrade systematically after 200 steps. Its quantized momentum is a storage optimization with negligible impact on gradient quality.

---

## Q3

**Answer: A**

**Why:** Held-out means held out — the key guarantee is that these examples were never used in training, not that they are stylistically different. The colleague is confusing distribution similarity (expected and not harmful) with data leakage (catastrophically harmful). A synthetic test set generated from the same pipeline as training data is a valid eval set as long as the test examples are unique and unseen. The right follow-up is to verify deduplication, not to dismiss the 71% number.

**Why others are wrong:**
- B: Synthetic test sets are not inherently worse than human-annotated ones for execution-based eval — what matters is that the SQL is correct and the test schemas are valid, which synthetic generation can achieve.
- C: Different random seeds do not make the data structurally independent if the templates and schema patterns are the same — this is a weaker argument than A.
- D: Execution-based eval is not immune to dataset bias. If the test examples only cover simple single-table SELECTs, a model scoring 71% there might fail on multi-join queries — the bias is in test coverage, not the eval mechanism.

---

## Q4

**Answer: A**

**Why:** Adapter weights are mathematically a linear transformation applied on top of the frozen base model. They are derived from the base in the sense that they are optimized to work with it, and distribution of the adapter is effectively distributing a modification of the base model's behavior. Qwen's Apache 2.0 license permits redistribution and derivative works under the same license — using Apache 2.0 for the adapter is both legally correct and the standard practice in the open-source ML community.

**Why others are wrong:**
- B: While the legal status of adapter weights is debated, the practical and safe choice is to follow the base model's license. Claiming full independence is legally risky.
- C: Apache 2.0 explicitly permits redistribution of derivative works — the license does not restrict this.
- D: CC-BY-4.0 is designed for creative works, not software or model weights. Using it for a model artifact creates legal ambiguity that Apache 2.0 avoids.

---

## Q5

**Answer: B**

**Why:** DoRA separates the weight update into a magnitude component (a scalar per output dimension) and a direction component (the LoRA low-rank subspace). This separation helps when the model's adaptation is directionally correct (it is generating SQL with the right structure) but incorrectly calibrated in magnitude (it underweights or overweights certain output distributions). If exec correctness is improving slowly while loss is stable, the model may be stuck in a good direction but wrong scaling — DoRA's explicit magnitude learning directly addresses this.

**Why others are wrong:**
- A: Parameter efficiency is not DoRA's primary advantage — standard LoRA is already parameter-efficient. DoRA adds a small number of magnitude scalars, which slightly increases parameter count.
- C: DoRA is typically 5–10% slower than standard LoRA due to the additional magnitude normalization step; it does not reduce training time.
- D: There is no compatibility requirement — DoRA adapters work on any base model regardless of how the base was pre-trained.

---

## Q6

**Model answer:** Complete inference pipeline for `postgres-sqlcoder-7b-v1`:

1. Input collection: user provides (a) natural language question and (b) database schema as CREATE TABLE statements. Application layer is responsible.
2. Prompt construction: format the inputs using the training prompt template — `### Schema:\n{schema}\n\n### Question:\n{question}\n\n### SQL:\n`. Tokenizer converts this to token IDs.
3. Model loading: NF4-quantized `Qwen/Qwen2.5-Coder-7B` base loaded via BitsAndBytesConfig; `postgres-sqlcoder-7b-v1` LoRA adapter loaded on top via `PeftModel.from_pretrained`.
4. Forward pass + generation: `model.generate` runs autoregressive decoding (greedy or beam search) to produce SQL tokens up to the EOS token or 256-token limit.
5. SQL extraction: tokenizer decodes the new tokens; any markdown fences (` ```sql `) are stripped.
6. Safety check: `is_safe_sql` confirms the output is a SELECT statement.
7. Execution: psycopg2 cursor executes the SQL against the target Postgres database with a 5-second statement timeout.
8. Result delivery: rows returned to the application layer for display or further processing.

---

## Q7

**Ranked hypotheses for the loss spike at step 150:**

1. (Most likely) A bad batch — a batch containing malformed or outlier training examples with very high loss caused a large gradient update that temporarily destabilized the optimizer's momentum states. Recovery by step 400 confirms the optimizer corrected itself.
2. Learning rate warmup ending — if the LR scheduler ends its warmup phase around step 150 and the peak LR is slightly too high, the first full-LR update can cause a spike before the model re-stabilizes.
3. Gradient accumulation boundary — if you use gradient accumulation of 4 and the effective batch at step 150 happened to aggregate several high-loss examples, the combined gradient was unusually large. This is a sub-case of hypothesis 1.
4. (Least likely) Checkpoint loading artifact — if you resumed training from a checkpoint at step 150 and the optimizer state was not correctly restored, the first step after resume can spike. Check your training logs for any resume events near step 150.

---

## Q8

**Model answer:** A model card that only contains training config is incomplete for any model, but especially for a text-to-SQL adapter. Users need: the exact prompt format (wrong format = garbage output — and the format is not derivable from the training config alone), the evaluation results on an independent test set (without this, users have no baseline to compare against their own use case), and the known limitations from error analysis (which SQL patterns fail). For a text-to-SQL model specifically, limitations matter because a production system might route queries to a fallback (GPT-4o API) when the local model is likely to fail — that routing logic requires knowing the failure modes. A training config alone tells you how the model was made, not whether or how to use it safely.

---

## Q9

**Model answer:**

(a) The comparison is misleading for three reasons. First, GPT-4o's 82% is zero-shot — it has seen SQL in pre-training and is using general reasoning. Your 71% is a fine-tuned 7B model, which is 20x smaller. The right comparison is 71% (your model) vs. 35% (base Qwen2.5-Coder-7B zero-shot) — a 36-point improvement from fine-tuning. Second, the test set is 100 examples from your synthetic domain; GPT-4o's broader training may advantage it on structural SQL knowledge while your model has domain-specific schema awareness that would show up in a larger, more diverse test. Third, Phase 4 was about building the pipeline and infrastructure — the eval harness, the dataset, the deployment artifacts — not maximizing the first model's accuracy.

(b) Two production advantages of a fine-tuned local model over GPT-4o: (1) Latency and cost — a local 7B model serves requests in 100–300ms on a GPU instance at $0.50/hr, versus GPT-4o API calls at $15–60/M tokens with 1–3 second round-trip latency. For a product serving 10K queries/day, cost savings are significant. (2) Data privacy — your PostgreSQL schemas and query history never leave your infrastructure. GPT-4o requires sending schemas to OpenAI's API, which is unacceptable for sensitive enterprise databases.

(c) The Phase 5 technique most likely to close the gap is GRPO with the execution harness as the reward signal. GRPO samples multiple SQL candidates per question, executes each, and rewards the model for generating correct SQL. Crucially, it learns from both good and bad samples — the model sees what it generated, what was wrong, and updates to prefer correct generation. This directly optimizes execution correctness rather than token imitation of training examples. Published results on similar text-to-SQL tasks show GRPO pushing 7B models from the 70–75% SFT plateau to 83–88%, which would match or exceed the GPT-4o baseline.
