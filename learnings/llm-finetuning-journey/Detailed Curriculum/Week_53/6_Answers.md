# Week 53 Answers

## Q1 — Answer: A

**Why:** Pre-training on hundreds of billions of tokens already teaches the model factual knowledge, language understanding, and even SQL syntax. Alignment fine-tuning teaches it how to respond — tone, format, when to say "I don't know." LIMA's 1,000 examples are sufficient because they need only calibrate the output style, not re-teach the underlying knowledge.

**Why B is wrong:** Catastrophic forgetting is real but not the primary mechanism LIMA exploits. A 1,000-example run is short enough to avoid severe forgetting regardless of quality.

**Why C is wrong:** The comparison in the paper controls for learning rate. The effect holds across hyperparameter configurations.

**Why D is wrong:** LIMA uses standard prompt/completion format. The gain is from curation, not format.

---

## Q2 — Answer: B

**Why:** Blindly deleting all members of duplicate pairs (A) is too aggressive — you lose good examples just because they resemble another. The correct strategy is to keep the highest-quality representative from each cluster. AST depth is a reasonable proxy for complexity, but you could also use a human judge or your Week 55 LLM-as-judge to pick the better member.

**Why A is wrong:** Deleting both members of a pair causes unnecessary data loss.

**Why C is wrong:** Near-duplicates do not provide useful augmentation — they cause the model to memorize phrasings rather than generalize.

**Why D is wrong:** Lowering the threshold catches more near-duplicates, not fewer. At 0.70, you would remove too aggressively.

---

## Q3 — Answer: C

**Why:** Contamination matters only when the contaminated examples appear in your evaluation set. If you evaluate on Spider's test split and your training data contains paraphrases of Spider test questions, your Spider score is artificially inflated. If you evaluate on your own custom benchmark only, Spider training set overlap is irrelevant. The practical rule: exclude overlap with every benchmark whose test set you will report results on.

**Why A is wrong:** Spider training set examples are legitimately usable for training — they are public training data.

**Why B is misses nuance:** "Only test set contamination matters" is correct in principle but requires you to know which benchmarks you will evaluate on.

---

## Q4 — Answer: C

**Why:** This is the highest-risk failure mode. A generic teacher prompt like "generate a SQL query for a PostgreSQL database" will produce mostly simple SELECT/WHERE/JOIN examples because those dominate the teacher's pretraining distribution. TimescaleDB hyperfunctions (time_bucket_gapfill, locf, interpolate) are rare in internet text. Without explicit, detailed prompts for these skills, your 30K synthetic examples will be heavily skewed toward basic SQL, defeating the purpose of v3.

**Why A is a real risk** but lower priority — you can post-process with `sqlglot` to convert dialects.

**Why B is wrong:** Generation cost is manageable and not the critical risk.

**Why D is wrong:** HuggingFace has no such limit.

---

## Q5 — Model Answer

412 examples representing ~2.2% of v2 is almost certainly insufficient if TimescaleDB performance is your primary differentiator. The rule of thumb in SFT research is that a skill needs roughly 500–1,000 examples to show reliable improvement over the base model, and 2,000+ to approach ceiling performance on that skill. At 412 examples, your model will have absorbed TimescaleDB syntax but likely cannot generalize to novel schemas or query patterns. The target for v3 should be at least 3,000 TimescaleDB-specific examples (covering all hyperfunction families, continuous aggregate syntax, compression, data retention policies, and combination queries). This is a 7× increase — achievable via targeted synthetic generation in Week 54 with carefully engineered teacher prompts.

---

## Q6 — Model Answer

Execution correctness means the query runs without a Postgres error and returns rows. Semantic correctness means those rows are the right rows for the natural language question asked.

Concrete failure case: Suppose the question is "Find the average temperature per hour for sensor_id 42 over the last 7 days." A semantically wrong but execution-correct answer:

```sql
SELECT time_bucket('1 hour', time) AS hour, AVG(temperature)
FROM sensor_readings
WHERE sensor_id = 42
GROUP BY hour;
```

This runs without error and returns rows. But it is semantically wrong: it computes the average over all time, not just the last 7 days. The correct query adds `AND time > NOW() - INTERVAL '7 days'`. The execution-correct filter would pass this wrong answer into your training data.

---

## Q7 — Model Answer

Three failure modes the execution-only criterion misses:

1. **Wrong rows, right structure.** As shown above — the query executes but returns incorrect data. Requires a reference answer comparison, not just execution.

2. **Schema-specific hacks.** The query may hardcode a specific value (e.g., `WHERE sensor_id IN (1,2,3,42)`) that happens to work on your test database but is not a valid general answer to the question. Execution passes; generalization fails.

3. **Inefficient but correct SQL.** A query might scan 100M rows with a full table scan when an indexed time-bucket query would be 100× faster. Both execute correctly. Your model trained on the slow version will generate production-hostile SQL. You need a quality signal beyond execution.

---

## Q8 — Model Answer

Target allocation for 30K examples across 15 skills with difficulty stratification:

First, reserve 40% of budget (12K examples) for your top 4 gaps (TimescaleDB-specific: 4K, window functions: 3K, CTEs with complex joins: 3K, multi-schema federation: 2K). Distribute the remaining 18K across 11 skills weighted by their gap size from the audit.

For skills where the teacher performs poorly (TimescaleDB hyperfunctions): generate with explicit few-shot examples in every prompt — include 2–3 working TimescaleDB queries as context. Run execution validation immediately after each batch. If execution rate falls below 60% for a skill, switch strategy: hand-write 50 verified examples, then use the teacher to augment/paraphrase them rather than generate from scratch.

Difficulty stratification: within each skill allocation, explicitly request Easy/Medium/Hard/Expert examples in 15/35/35/15 ratio. For Expert, always include the schema DDL in the teacher prompt and require the query to use at least 2 advanced constructs.

Fallback if < 60% execution rate persists: reduce allocation for that skill by 50%, spend the saved budget on a human-verified hand-written set of 100 examples for that skill, and use those as few-shot context for subsequent teacher calls.
