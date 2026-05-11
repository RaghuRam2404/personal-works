# Week 26 Quiz Answers

## Q1 — Answer: B

**Answer:** B — Keep both; they are below the 0.7 threshold.

**Why:** MinHash LSH with threshold=0.7 only flags pairs as near-duplicates if their estimated Jaccard similarity exceeds 0.7. The two questions "Find all orders from last week" and "List all orders placed in the previous 7 days" have a Jaccard similarity of 0.65 on 5-word shingles — semantically similar but different enough to survive the filter. This is acceptable: slightly similar questions with the same SQL teach the model that multiple phrasings map to the same query, which is valuable training signal.

**Why others are wrong:**
- A: the threshold is 0.7; 0.65 is below the threshold, so they are not marked as duplicates
- C: MinHash uses Jaccard similarity on n-gram sets, not semantic similarity; two paraphrases could score low Jaccard if they use different words
- D: MinHash does not merge examples; it only identifies near-duplicates for removal

---

## Q2 — Answer: B

**Answer:** B — Only you (and explicitly granted collaborators) can see and download the dataset.

**Why:** HuggingFace Hub's `private=True` flag restricts visibility to the dataset owner and any collaborators you explicitly invite. The dataset does not appear in search results, cannot be discovered by other users, and requires authentication to access. This is appropriate for your domain dataset which may contain proprietary schema structures.

**Why others are wrong:**
- A: datasets are not encrypted on HuggingFace; access is controlled by permissions, not encryption
- C: the license field in the dataset card controls commercial use, not the private flag
- D: private datasets persist indefinitely on HuggingFace Hub

---

## Q3 — Answer: C

**Answer:** C — Executing against a real PostgreSQL database with realistic test data.

**Why:** sqlglot parse verifies syntax (confirms the SQL is parseable) but not semantics — a syntactically valid query can return wrong results, reference wrong columns, or produce incorrect aggregations. GPT-4 review is helpful but fallible — LLMs make SQL mistakes. The only true verification is execution: create the tables, insert representative data, run the query, and check that the returned rows match the question's intent. This is why the Curriculum specifies: "verify every SQL with sqlglot AND execute against a local Postgres instance."

**Why others are wrong:**
- A: sqlglot parse is necessary but insufficient for semantic correctness
- B: GPT-4 is helpful for review but makes mistakes, especially with complex temporal queries and TimescaleDB functions
- D: BLEU measures textual similarity to a reference, not correctness

---

## Q4 — Answer: A

**Answer:** A — The model will overuse this pattern, applying it when a different interval is appropriate.

**Why:** Neural language models learn from the training distribution. If 30% of Tier 3 examples use `time_bucket('1 hour', ts)` specifically, the model will assign high probability to generating exactly `'1 hour'` whenever it generates a `time_bucket()` call, even when the question asks for daily, weekly, or monthly aggregates. This is a systematic bias caused by insufficient diversity in the generated examples. Fix: add diverse `time_bucket` examples with all common intervals ('15 minutes', '6 hours', '1 day', '1 week', '1 month') in roughly equal proportion.

**Why others are wrong:**
- B: `time_bucket` is valid for any interval, including hours
- C: MinHash deduplication operates on question text, not SQL patterns; it would not remove all `time_bucket('1 hour')` queries if the questions are sufficiently different
- D: repetitive patterns cause a specific bias toward that pattern, not catastrophic forgetting of other skills

---

## Q5 — Answer: C

**Answer:** C — Both A and B are valid approaches depending on criticality.

**Why:** If `LATERAL` joins are frequently needed in your deployment (e.g., for running-total queries or correlated lookups in your PostgreSQL application), then under-representation in training data is a real problem — the model will rarely generate them. Up-weighting (repeating) the 20 LATERAL examples in the training data biases the model toward generating them more often. If `LATERAL` joins are rare edge cases in practice, accepting their low generation frequency is appropriate. The correct answer depends on deployment requirements: critical → up-weight; rare → accept.

**Why others are wrong:**
- A is correct in some circumstances but not universally
- B is correct in some circumstances but not universally
- D is wrong: even 20 examples with enough repetition can teach a pattern; zero examples guarantees the model never generates it

---

## Q6 — Short Answer

Cross-deduplication ensures that synthetic (Tier 3) examples do not repeat questions that already exist in Tier 1 (Spider/BIRD) or Tier 2 (hand-written). Without it, the same question might appear multiple times in the dataset — once from Spider and once from Self-Instruct generation. This has two failure modes: (1) duplicate examples inflate validation accuracy because the model has "memorized" those specific questions, making evaluation metrics unreliable; and (2) the training signal for those questions is over-weighted relative to others, causing the model to learn those patterns disproportionately strongly while being weak on under-represented query types.

---

## Q7 — Short Answer

**Problem:** An average SQL length of 18 tokens and 40% trivial single-table queries means your dataset is skewed toward simple queries. A model fine-tuned on this distribution will default to generating simple SQL even for complex questions, and will rarely produce JOINs, CTEs, or multi-step aggregations.

**Fix 1: Quality filter on complexity.** Add a filter that rejects SQL with fewer than 10 tokens, or that contains neither a JOIN, CTE, subquery, window function, nor aggregation. Apply this retroactively to Tier 3 (re-generate rejected examples). Accept that Tier 1 Spider examples often have simple queries — this is intentional for baseline coverage.

**Fix 2: Category-targeted self-instruct.** Re-run Self-Instruct with generation prompts that specifically require complexity: "Generate 20 questions that require a SQL query using at least two JOINs and a GROUP BY clause." Use separate generation batches for different complexity categories (CTEs, window functions, nested subqueries) and ensure each category has at least 5% representation in the final dataset.

---

## Q8 — Dataset Evaluation Protocol (3 checks)

**Check 1: SQL Validity Rate**
- Run all 5,000 examples through `sqlglot.parse(sql, dialect="postgres")`
- Pass: ≥ 98% of examples parse without error
- Failure action: investigate failing examples; likely caused by self-instruct generation with incomplete schema conditioning

**Check 2: Schema Consistency Rate**
- For a sample of 200 examples, manually verify that every table and column referenced in the SQL appears in the schema provided in the user message
- Pass: ≥ 90% of examples have consistent table/column references
- Failure action: the schema generation in your Self-Instruct prompt is too vague; add schema explicitly to generation prompts

**Check 3: Distribution Coverage Check**
- Count examples containing each of: JOIN, WITH (CTE), window function (OVER), GROUP BY, HAVING, TimescaleDB time_bucket, ON CONFLICT, RETURNING, JSONB operator
- Pass: each construct appears in ≥ 2% of examples (100 examples minimum)
- Failure action: for any construct below 2%, run targeted Self-Instruct generation to add 100+ examples of that construct

---

## Q9 — Scenario Model Answer

**1. The 85% vs 45% gap indicates overfitting to training question phrasing.** Your training dataset contains specific question phrasings and schema names that the model has partially memorized. When questions are rephrased (even slightly) or use schemas it has not seen, accuracy drops dramatically. This is a distribution mismatch problem: your training data does not have sufficient diversity in question phrasing and schema variation.

**2. `time_bucket_gapfill()` underrepresentation.** With only 3 examples of `time_bucket_gapfill()` in 5,000 training examples (0.06%), the model has essentially not learned this function. 30% of test failures = 15 test questions × 0.30 = ~5 failures all involving `time_bucket_gapfill`. Fix: add 50–100 examples specifically demonstrating `time_bucket_gapfill()` with gapfill semantics, including the `USING` clause and NULL handling. These become the highest-priority additions to v2.

**3. Semantic alias error: "total value" → wrong aggregation.** This is a world-knowledge gap: the model does not know that "total value" in an order context means `quantity × unit_price`, not just the value of a single column. The training data always had "total revenue" computed as `SUM(amount)` (a pre-computed column), so the model learned this shortcut. Fix: add training examples where "total value" or "order total" requires multi-column arithmetic (`SUM(quantity * unit_price)`), and include schema context that makes the correct formula obvious (`quantity INTEGER`, `unit_price NUMERIC`).

**4. postgres-sql-v2 plan:**
- Add 100 `time_bucket_gapfill()` examples (highest priority)
- Add 200 multi-column arithmetic aggregation examples
- Diversify schemas: ensure each schema appears in at most 5% of examples
- Add 300 paraphrase pairs: same question, different phrasing, same SQL
- Increase LATERAL join examples from 20 → 100
- Add 100 examples for each currently underrepresented construct
- Target: 8,000–10,000 examples in v2 with better coverage verification
