# Week 23 Quiz Answers

## Q1 — Answer: B

**Answer:** B — Log-likelihood scoring.

**Why:** For multiple-choice benchmarks, lm-eval concatenates each option with the question and computes the total log-likelihood of the option tokens given the question tokens. The model selects the option with the highest log-likelihood. This approach works for any causal language model without requiring instruction-following or the ability to generate "A/B/C/D."

**Why others are wrong:**
- A: generating the letter requires instruction tuning and specific prompt formatting; log-likelihood scoring does not
- C: embedding similarity requires a separate encoder; lm-eval uses the model's generative probability
- D: no separate classifier is involved; the base model's likelihood is used directly

---

## Q2 — Answer: B

**Answer:** B — Near random, which is expected for 50M parameters on MMLU.

**Why:** MMLU requires factual knowledge spanning 57 domains — law, medicine, mathematics, history, etc. A 50M parameter model does not have enough capacity to store this knowledge from a training corpus. GPT-3 (175B params) scores only 43% on MMLU. Expecting 27% from a 50M model is correct calibration.

**Why others are wrong:**
- A: near-random performance on MMLU at 50M params is expected, not a failure
- C: memorizing the test set would show up as near-perfect accuracy, not near-random
- D: random behavior IS possible; the model's log-likelihoods can be nearly equal across options

---

## Q3 — Answer: B

**Answer:** B — To prevent bias toward shorter options.

**Why:** Log-likelihood is cumulative — a 10-token option starts with a lower absolute log-probability than a 2-token option simply because there are more terms in the sum. Length normalization divides by the number of tokens, converting absolute log-likelihood to average per-token log-likelihood. This allows fair comparison across options of different lengths.

**Why others are wrong:**
- A: compute budget is not relevant here
- C: BPC conversion has nothing to do with option selection within a benchmark
- D: the normalization is over total option length, not penalizing first tokens specifically

---

## Q4 — Answer: B

**Answer:** B — The model learned surface statistics but not factual relationships.

**Why:** Perplexity measures how well the model predicts the next token in Wikipedia text. This can be achieved by learning strong n-gram statistics, discourse patterns, and word co-occurrence without understanding factual relationships. MMLU questions test whether the model knows that "the capital of Australia is Canberra" or "mitosis produces diploid cells" — facts that require correctly encoding and retrieving semantic relationships, not just predicting likely next words in context.

**Why others are wrong:**
- A: low perplexity and low MMLU accuracy is a well-documented phenomenon for domain-specialized or small models
- C: vocabulary mismatch would affect perplexity, not just MMLU
- D: the relationship between perplexity and MMLU is not a threshold function

---

## Q5 — Answer: B

**Answer:** B — Prepends 5 labeled example pairs to each test question as context.

**Why:** Few-shot evaluation provides the model with k examples of (question, correct answer) pairs as in-context demonstrations before the actual test question. The model sees the task format and uses in-context learning to infer the expected response style. This does not update any weights — it is purely inference-time conditioning.

**Why others are wrong:**
- A: no training occurs; weights are frozen
- C: multiple seeds are used for generation sampling, not for benchmark scoring
- D: multiple prompts per question is a different technique (self-consistency)

---

## Q6 — Short Answer

Benchmark contamination means the model's training data includes the test split of the benchmark, so the model has "seen the answers" during training. For open-weight models where training datasets are disclosed, this can be detected by checking if test questions appear verbatim in the training data using n-gram matching. For closed models, contamination is inferred by checking if the model scores anomalously high on certain subsets while underperforming on structurally similar questions that were not in the corpus. A specific detection method: compute model accuracy on the published test set and on a held-out "fresh" set of equivalent difficulty questions — a large gap (>10%) is suspicious.

---

## Q7 — Short Answer (50M scores 30% vs GPT-2 43% on ARC-Easy)

1. **Training token count:** GPT-2 was trained on 40B tokens (WebText); your model on ~2B tokens. More tokens provide better calibration of factual relationships. Science questions in ARC-Easy require the model to have encountered the relevant facts during training.

2. **Training data distribution:** GPT-2's WebText was curated from high-quality Reddit links, biasing toward informative factual content. FineWeb-Edu is educational quality, which helps, but 2B tokens is still much less exposure to STEM facts than GPT-2's 40B tokens of diverse high-quality text.

3. **Parameter capacity:** 117M vs. 56M parameters means GPT-2 has ~2× the capacity to store statistical patterns that map questions to correct answers. Even with equivalent training tokens, the smaller model has less room for the relevant factual associations.

---

## Q8 — Short Answer (PostgreSQL text-to-SQL evaluation suite)

**Benchmark 1 — Execution Accuracy (EA)**
- Metric: Fraction of generated queries that produce the correct result when executed against the reference database
- Dataset: Your held-out PostgreSQL/TimescaleDB examples (Week 26 dataset, 20% held out)
- Measures: End-to-end correctness including syntax, semantics, and query logic

**Benchmark 2 — Spider Accuracy**
- Metric: Exact match and execution match on the Spider test set
- Dataset: [Spider benchmark](https://yale-lily.github.io/spider)
- Measures: Generalization to unseen schemas and databases; Spider scores are comparable across the literature, enabling benchmarking against other models

**Benchmark 3 — TimescaleDB Extension Coverage**
- Metric: Fraction of generated queries that correctly use TimescaleDB-specific functions (time_bucket, compress_chunk, continuous aggregates)
- Dataset: Hand-crafted 50-question evaluation set of TimescaleDB-specific queries
- Measures: Domain-specific knowledge that Spider does not cover; directly relevant to your deployment scenario

---

## Q9 — Scenario Model Answer

**1. Why MMLU and HellaSwag are insufficient:**
MMLU measures factual knowledge across 57 general domains; HellaSwag measures narrative common sense. Neither benchmark contains SQL generation tasks, PostgreSQL syntax, or time-series database concepts. A model could score 80% on MMLU by memorizing medical and legal facts without being able to write a correct `SELECT` statement. The evaluation must match the deployment task.

**2. Primary metric:**
Execution Accuracy (EA): the percentage of generated SQL queries that execute successfully and return the same rows as the reference query when run against the test database. This is the ground truth metric — the SQL is either correct (returns the right data) or it is not. Syntactic similarity metrics like BLEU or exact match are insufficient because multiple SQL formulations can be semantically equivalent.

**3. Constructing a contamination-free eval set:**
Collect questions and SQL pairs from databases and schemas not present in any publicly released text-to-SQL dataset (Spider, BIRD, WikiSQL). Use your own PostgreSQL/TimescaleDB schemas from work or personal projects. Hand-write the questions and reference queries yourself. Do not use any SQL from GitHub that may have appeared in the model's training data (Qwen2.5-Coder was trained on The Stack v2).

**4. Why Spider 82% may not transfer:**
(a) Spider uses SQLite-style queries over generic schemas (users, products, flights). Your PostgreSQL/TimescaleDB benchmark uses PostgreSQL-specific syntax (RETURNING, ON CONFLICT, JSONB operations, time_bucket(), continuous aggregates). Models trained on Spider data optimize for generic SQL and often fail to generate valid PostgreSQL-specific constructs.
(b) Spider schemas are simple (3–5 tables, few foreign keys). TimescaleDB schemas in production have hypertables, partitioning, compression policies, and complex time-series joins. A model that handles Spider's simple joins may not generalize to 10-table time-series schemas with temporal predicates.
