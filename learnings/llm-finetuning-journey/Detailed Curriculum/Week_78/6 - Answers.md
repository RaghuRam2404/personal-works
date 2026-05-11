# Week 78 Answers — Final Phase 6 Gate and Course Wrap

---

## Q1. Answer: D

**Why D is correct:** Both actions are necessary and complementary. Publishing Custom-200 addresses the reproducibility concern directly — anyone can now evaluate their model against your benchmark and compare. Citing BIRD-SQL dev (68.4%) provides an externally verifiable anchor point that researchers and engineers can look up independently. One without the other is incomplete: BIRD-SQL alone does not capture your TimescaleDB advantage, and Custom-200 alone is unverifiable without publication. Both together give the recruiter and the community what they need.

**Why A alone is insufficient:** Publishing Custom-200 helps future evaluators but does not give the recruiter an immediate external reference point they can check today.

**Why B alone is insufficient:** 68.4% on BIRD-SQL dev is a strong result, but it does not explain or capture the TimescaleDB-specific advantage that the Custom-200 number represents.

**Why C is wrong:** A well-designed domain benchmark published alongside the model is valuable, not useless. The entire field of NLP depends on researchers publishing their benchmarks alongside their models. Conceding this point incorrectly devalues your work.

---

## Q2. Answer: B

**Why B is correct:** An effective limitation statement is specific, testable, and actionable. B identifies: (a) a concrete scope boundary (PostgreSQL/TimescaleDB only), (b) a specific example of what would likely fail (MySQL backtick quoting, LIMIT without OFFSET), and (c) the reason for the limitation (training data composition). A reader can act on this — they know not to use your model for MySQL without further evaluation.

**Why A is wrong:** "May produce incorrect SQL" is true of every model ever built. It conveys no useful information about when or why failures occur.

**Why C is wrong:** Comparing parameter count to GPT-4 is neither a meaningful nor an accurate framing. GPT-4 has more parameters but is a general model; your specialized 7B model outperforms it on your domain. Size is not the relevant dimension.

**Why D is wrong:** "Hallucination" as a limitation is similarly too vague to be actionable. Specific failure modes (wrong column names, missing WHERE filters on TimescaleDB-specific clauses) would be useful; a generic hallucination caveat is not.

---

## Q3. Answer: C

**Why C is correct:** This is the engineering-appropriate response. A separate Oracle-specific LoRA adapter (frozen base model, new adapter weights) is the cleanest way to add Oracle capability without interfering with PostgreSQL performance. A lower learning rate reduces the risk of the adapter overwriting shared SQL knowledge. The held-out Oracle eval set is mandatory — without it, you cannot claim the fine-tuning worked. Being transparent with the client about the preliminary nature of the evaluation is professional.

**Why A is wrong:** Oracle syntax differs from PostgreSQL in structural ways (ROWNUM vs LIMIT, CONNECT BY for hierarchical queries, different date functions). "SQL is SQL" is a category error — fine-tuning on 500 examples without evaluation will likely produce a model that sometimes generates Oracle syntax and sometimes PostgreSQL syntax, with no reliable dialect discrimination.

**Why B is wrong:** Declining is overly conservative. You can address the client's need with a careful adapter-based approach and honest evaluation. Refusing to try is not good engineering.

**Why D is wrong:** Recommending a competitor's product without attempting a well-designed solution first abandons the client prematurely and ignores your actual capability.

---

## Q4. Answer: B

**Why B is correct:** This is a proper ablation study design. To understand which stage contributed what, you must isolate each stage by comparing adjacent checkpoints: (CPT-only vs SFT-v3) measures the SFT contribution, (SFT-v3 vs DPO-v3) measures the DPO contribution, (DPO-v3 vs GRPO-final) measures GRPO's contribution. Running all these comparisons on the same benchmark (Custom-200) with the same evaluation script gives you an additive attribution of performance gains across stages.

**Why A is wrong:** Comparing SFT-v3 to GRPO-final conflates the contributions of DPO and GRPO together. You cannot attribute the improvement to a single stage.

**Why C is wrong:** Running GPT-4o on your training data measures GPT-4o's in-distribution generalization, not your pipeline's stage contributions. This is the wrong experiment.

**Why D is wrong:** This is one valid pairwise comparison, but it only isolates GRPO's contribution. You need all pairwise comparisons to understand the full pipeline.

---

## Q5 — Short Answer

**Weakness 1: Benchmark contamination risk.** Custom-200 was created by you, based on your domain knowledge and your schema. It is possible (even likely) that the query patterns you chose to evaluate on overlap significantly with the patterns you chose to train on — not because you deliberately cheated, but because both came from your domain knowledge and intuition. This is evaluation data leakage through the designer's mind. To address this rigorously, you would need an independent evaluator to construct the test set without access to the training data, or you would need to use a stratified sample from a corpus of real user queries that neither you nor your training pipeline had access to during development.

**Weakness 2: Limited size and diversity.** 200 examples is small. Statistical variability is high: a 95% confidence interval on 83.1% accuracy with n=200 is approximately ±5.2%, meaning the true accuracy could be anywhere from 78% to 88%. Additionally, if the 200 examples are not stratified across query types, a single category can dominate the aggregate score. To address this: expand to 1,000+ examples, stratify by query type (single-table, multi-table JOIN, aggregation, time-series, CTE), and report per-stratum accuracy rather than a single aggregate number.

---

## Q6 — Short Answer

PPO (Proximal Policy Optimization) is an actor-critic method: it trains both a policy model and a separate value (critic) model to estimate expected reward, and uses the value model to compute advantage estimates that stabilize policy updates. GRPO (Group Relative Policy Optimization) eliminates the value model entirely by generating K responses per prompt, ranking them by reward, and computing relative advantages within the group — the K responses serve as a Monte Carlo estimate of the value baseline.

GRPO was the right choice for your setup for two reasons. First, maintaining a value model for a 7B parameter policy requires loading a second 7B parameter network — doubling GPU memory requirements during RLHF training, which is prohibitive on RunPod consumer GPU instances. GRPO's group-sampling baseline eliminates this cost entirely. Second, your reward signal is binary or near-binary (SQL executes correctly or not, possibly with a soft partial-credit reward), which makes the relative ranking within a group of K responses a reliable advantage estimator even without a learned value function.

GRPO's limitation compared to PPO: the group-sampling advantage estimate has higher variance than a learned value function, especially on sparse rewards where most of the K responses receive the same reward (all wrong or all correct). In practice this means GRPO can be noisy for prompts at the extremes of difficulty. For your setup with K=8 and a mixed-difficulty prompt set, this variance was manageable.

---

## Q7 — Short Answer

The "curse of task specificity" refers to the trade-off inherent in any domain-specialized fine-tuning: the more you optimize a model for one narrow task or dialect, the more you risk degrading its performance on adjacent but distinct tasks. Your capstone model is a clear example: you fine-tuned on PostgreSQL and TimescaleDB queries, so the model is highly calibrated for that dialect and those function names. On WikiSQL (a simpler, different-schema benchmark) or Oracle-dialect queries, you should expect meaningful performance degradation relative to your TimescaleDB EM.

From a research perspective, this is a genuine problem. If you want to publish your model as a general NL→SQL contribution, you need results on standard benchmarks (Spider, BIRD-SQL, WikiSQL) and your TimescaleDB-specific advantage must be shown to not come at the cost of general capability. From a product perspective, this is not a problem — it is a feature. Your customers are building PostgreSQL and TimescaleDB applications. They do not need your model to work on Oracle. A 83.1% EM on their actual queries is what matters, not generality. The curse of task specificity is only a curse if generality is a requirement; for focused enterprise deployment, specificity is precisely the point of fine-tuning.

---

## Q8 — Deep Scenario

**Model answer:**

**Data challenge — schema drift:** In a production BI tool, schemas change constantly: new columns added, tables renamed, foreign keys modified. Your training data contains fixed schemas, so the model's SQL generation degrades when it encounters column names or table structures not seen during training. The real-world consequence is silent failures — the model generates syntactically valid SQL that references non-existent columns, returning no rows or wrong rows without any error. The mitigation you implemented is schema-aware prompting (Week 53): injecting the full current schema into every prompt so the model always has access to the live column list. For production, you would extend this with retrieval-augmented schema selection (only inject the top-K most relevant tables for each question) to stay within context limits as schemas grow.

**Evaluation challenge — ground truth annotation:** In your course, you had gold-standard SQL for Custom-200 from known correct queries. In production, users ask questions no one has pre-labeled. Collecting ground truth at scale requires either hiring SQL experts to annotate user queries or implementing an execution-based pseudo-labeling system (execute the model's SQL and a reference query, compare result sets). The consequence if unaddressed is that you cannot measure model quality degradation in production until users complain. From Week 55 onward you built an execution-based evaluation pipeline; in production you would deploy this as an online monitoring system that samples 5% of live queries, executes them, and computes result-set match against a reference query when one exists.

**Deployment challenge — latency at inference:** Your vLLM deployment (Week 66) handles batch inference efficiently, but a BI tool has strict latency SLAs: a business user clicking "Ask a question" expects a response in under 2 seconds. A 7B model generating 200 tokens of SQL takes 3–5 seconds on a single A10G GPU without optimization. The mitigation is two-pronged: first, use speculative decoding or the GGUF Q4_K_M quantized model for the fast path (Week 63–64 work), which reduces generation time to under 2 seconds; second, implement a query cache that serves cached SQL for semantically similar questions using embedding similarity, bypassing generation entirely for repeated patterns (which in BI tools account for 40–60% of traffic).

---

## Q9 — Deep Scenario

**Model answer:**

The claim is partially supported but overstated. Your results do demonstrate that a 7B model, fine-tuned on domain-specific data with a careful multi-stage pipeline, can outperform frontier general-purpose models on a domain benchmark. This is an important proof of concept: it shows that for well-scoped, data-rich domains, compute-efficient specialization can substitute for scale. The key conditions that made this possible in your case were: (1) the domain is relatively narrow (one database dialect, one schema family), (2) you had or could generate sufficient high-quality training data (25K SFT examples, 5K DPO pairs), and (3) exact match on SQL is a deterministic, automatable metric that enabled scalable evaluation and reward signal.

The claim misses three important caveats. First, "the right approach" implies exclusivity — but the comparison to GPT-4o uses GPT-4o with a generic prompt, not GPT-4o with the same schema-aware prompting, few-shot examples, and domain context that your model was trained on. A fair comparison would apply chain-of-thought prompting and domain-specific context to GPT-4o as well, and the gap might narrow substantially. Second, the approach only generalizes to domains where training data is abundant and collectible — for rare or highly specialized domains (legal contracts, proprietary enterprise software), neither you nor most organizations can assemble 25K labeled examples, and the claim breaks down. Third, the total cost of the fine-tuning pipeline (compute, data curation, evaluation engineering) must be compared honestly against the marginal cost of GPT-4o API calls with a well-engineered prompt — for low-query-volume applications, the API may be more cost-effective even at lower accuracy. The claim is a useful heuristic for high-volume, well-defined domains; it is not a universal principle.
