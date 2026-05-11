# Week 61 Answers

## Q1 — Answer: D

**Why:** Each framing illuminates a different aspect. Absolute gap (9pp) is the most common and fair framing for reporting capability difference. Relative error rate is useful when comparing to human performance. The domain benchmark comparison is important for your core claim — your model specializes in PostgreSQL/TimescaleDB, not in BIRD's general business SQL. A complete technical report would present all three framings with appropriate caveats about benchmark relevance to the target use case.

---

## Q2 — Answer: B

**Why:** Your 95% CI of [63%, 73%] overlaps the paper's reported 70%. You cannot conclude your model is better or worse than the paper's model with statistical significance. This is the correct conclusion: "The results are not statistically distinguishable from [Paper X]'s reported 70%." Note that papers without confidence intervals often appear to have precise numbers but actually have substantial uncertainty. Reporting your CI is more scientifically rigorous, not less.

---

## Q3 — Answer: B

**Why:** For a question asking for readings "from the last 24 hours," `LIMIT 100` without the time filter is semantically wrong. If the database has more than 100 readings from the last 24 hours, the LIMIT-only query returns the right rows. But if the database has older data, it may return readings from days ago. The result sets are different (except on a database that happens to have exactly 100 recent readings). Execution accuracy with result comparison correctly marks this as incorrect because the result sets differ on the actual test database.

---

## Q4 — Answer: B

**Why:** Your training data (PostgreSQL/TimescaleDB enterprise queries, Stack Overflow Q&A about PostgreSQL, GitHub migration files) is stylistically closer to Defog's enterprise SQL distribution than to BIRD's academic benchmark questions. BIRD questions are crafted to test specific SQL reasoning capabilities in controlled ways; Defog questions reflect real users' natural language and real enterprise schemas. Your model has seen more Defog-style questions during training.

---

## Q5 — Model Answer

Test set contamination via model selection (also called "indirect data snooping"): even if you don't train on test set examples, selecting your final model based on test set performance means your reported test score reflects the best result from multiple experiments that all happened to see the test set. The reported score is therefore optimistic — it is the maximum across your model variants, not an unbiased estimate.

The correct procedure: use the dev set for all model selection, hyperparameter tuning, and comparison. Report the test set score exactly once, on your final chosen model, without the ability to revise it. If BIRD or Spider test set labels are not public (they aren't for most of these benchmarks), then your leaderboard submission becomes your one-shot test set evaluation.

For your project: use the dev sets for selecting between SFT-v3, DPO-v3, and GRPO-final. Your custom 200-example benchmark (never seen during any training stage) serves as your "test set" for honest final reporting.

---

## Q6 — Model Answer

Execution accuracy with result comparison overestimates capability for the 15% of "right-looking but wrong" cases. These are queries that:
- Return the same result on the specific test database but would return different results on a different database instance
- Are semantically wrong (e.g., different time window) but happen to return the same rows because the test database only has data within the relevant time range
- Use a different aggregation that yields the same value on the specific data

Execution accuracy underestimates capability for cases where the model generates an equivalent but stylistically different correct query (e.g., a subquery instead of a JOIN that produces identical results). Both queries are correct but only exact match would count one as wrong.

Net effect: execution accuracy is a noisy but useful metric. It is better than exact match for most practical purposes. For honest reporting, supplement it with: (a) human review of a random 50-question sample to estimate the "right-looking-wrong" rate, and (b) a note in your report that execution accuracy may be slightly biased in either direction.

---

## Q7 — Model Answer

**Simple questions (78%):** The 22% failure rate on simple questions indicates: (a) prompt format mismatch — some simple questions use schema elements your model misidentifies; (b) occasional hallucinated column names for databases the model hasn't seen before. Fix: add more diverse schema examples in training, especially simple 1–2 table schemas with unusual column names.

**Moderate questions (64%):** The drop from 78% to 64% is mostly driven by multi-table joins with aggregation. Your model joins tables correctly in isolation but fails when aggregation and join need to be combined in specific orders. Fix: add training examples that specifically combine JOIN + GROUP BY + HAVING in diverse patterns; ensure your v3 dataset has at least 2,000 medium-difficulty examples with this pattern.

**Challenging questions (41%):** Three sub-categories:
- Multi-table CTEs with window functions: your model likely generates valid CTEs but has errors in the window frame specification (missing ROWS BETWEEN / RANGE BETWEEN). Fix: targeted training on CTE + window function combinations.
- Queries requiring external knowledge: BIRD provides context in the question text (e.g., "risk score > 0.7 indicates high risk"). Your model needs to extract and use this context. Fix: add instruction-following examples that extract numeric thresholds from the question text and embed them in SQL.
- Implicit schema relationships: BIRD questions that say "find customers who are also employees" when there's no explicit foreign key between customers and employees — you need to infer the relationship from column names. Fix: training examples that require semantic schema inference from column name patterns.
