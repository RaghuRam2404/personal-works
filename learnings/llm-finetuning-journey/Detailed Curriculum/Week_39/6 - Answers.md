# Week 39 Quiz Answers

## Q1

**Answer: B**

**Why:** SQL is not a unique representation of intent. Aliases, table prefix variations, equivalent JOIN orderings, and equivalent WHERE rewritings all produce the same result set. Exact match penalizes any textual difference even when the semantics are identical. Execution-based eval sees the result set, not the token sequence, so it correctly accepts all semantically equivalent variants.

**Why others are wrong:**
- A: Memorization would inflate exact match, not execution correctness. If anything, a memorized model would score higher on exact match.
- C: Execution-based eval compares full result sets (all values, all rows), not just row counts. It is more demanding than row count comparison.
- D: A whitespace bug would affect exact match in the other direction — fixing it would raise exact match, narrowing the gap.

---

## Q2

**Answer: B**

**Why:** If test example N inserts rows and the harness fails to clean them up (due to an exception mid-run), those rows persist in the shared schema. Test example N+1 then executes against a non-empty database that does not match the intended state. This is a classic test isolation failure. `DROP SCHEMA ... CASCADE` atomically removes all objects and is the safest approach.

**Why others are wrong:**
- A: DROP TABLE and DROP SCHEMA have similar performance characteristics for small schemas. Speed is not the concern.
- C: PostgreSQL fully supports dropping individual tables in any schema.
- D: The connection can be reused across schema drops; reconnection is not required.

---

## Q3

**Answer: C**

**Why:** A cross-join or a large hash join without an index can trigger a sequential scan across both tables. On 500K rows this can easily exceed 5 seconds. The `statement_timeout` will fire and Postgres will raise an exception, which your harness catches as an execution error. This is correct behavior — a model that generates a query requiring a full table scan on large tables is generating impractical SQL.

**Why others are wrong:**
- A: 500K rows is not automatically fast; it depends entirely on the query plan and whether indexes exist.
- B: `SELECT *` is still a SELECT statement; `is_safe_sql` would accept it.
- D: `statement_timeout` applies to all statement types including SELECT.

---

## Q4

**Answer: C**

**Why:** The NULL-to-string normalization is a pragmatic approximation. It fails only if a text column could legitimately contain the four-character string "NULL". In most SQL evaluation benchmarks this does not occur because schemas use typed numeric or short text fields. However, it is an important limitation to document. The safer approach is to use a dedicated Python sentinel: `_NULL = object()` and map SQL NULL to `_NULL` — but this requires a custom comparison function that handles the sentinel type.

**Why others are wrong:**
- A: While rare, a VARCHAR column could legitimately store the string "NULL" — the claim that it "can never" happen is too strong.
- B: A `object()` sentinel works but is not the only correct approach; documenting the limitation and accepting the approximation is also valid engineering in a controlled benchmark.
- D: Python's `==` does not have special handling for database NULLs. `None == None` is `True` in Python (unlike SQL), so technically Python handles it fine — but the issue is type coercion across DB drivers, not `None` comparison.

---

## Q5

**Answer: B**

**Why:** The model already executes successfully 94% of the time — syntax is not the bottleneck. The 36-point gap between success (94%) and correctness (58%) is entirely semantic: the SQL runs but returns wrong rows. The right fix is to understand why — wrong WHERE clauses, wrong aggregations, wrong joins — and add targeted training data. Broad interventions like increasing rank or switching to full fine-tuning address capacity, not data coverage.

**Why others are wrong:**
- A: Fixing the remaining 6% syntax errors gets you to 94% correctness at best — nowhere near closing the 36-point gap.
- C: LoRA rank affects the model's ability to learn transformations, but for a 7B model on 15K examples, rank 16 is already sufficient. The problem is training data distribution, not model capacity.
- D: Quantization does not materially affect semantic understanding at NF4 precision for a 7B model.

---

## Q6

**Model answer:** Execution correctness measures the actual semantic output of the SQL — the rows returned — rather than the textual form of the query. Two queries can be textually different but semantically identical, so exact match undercounts correct predictions. Conversely, a query can match exactly but fail to execute due to a typo, which exact match would count as correct and execution would catch. Concrete example: the expected SQL is `SELECT name FROM employees WHERE dept = 'eng'` and the model generates `SELECT e.name FROM employees AS e WHERE e.dept = 'eng'`. Exact match returns 0 (different tokens), execution correctness returns 1 (identical result set).

---

## Q7

**Model answer:** A CTE (`WITH cte AS (SELECT ...) SELECT * FROM cte`) is syntactically a SELECT statement at the outer level. The `sqlparse` library parses the statement type from the first keyword of the outermost statement — if the query begins with `WITH`, `sqlparse` may return `None` or `"UNKNOWN"` as the type rather than `"SELECT"`. This would cause `is_safe_sql` to reject a perfectly safe query. The fix is to check whether the statement contains only SELECT-family keywords: strip the CTE prefix and check the inner query type, or use `sqlparse.sql.Statement.get_type()` on the inner SELECT portion. Alternatively, you can parse the full statement and check that no DML tokens (INSERT, UPDATE, DELETE, DROP, CREATE) appear anywhere in the token tree.

---

## Q8

**Model answer:** Three specific interventions for wrong-filter failures:

1. Add training examples with complex WHERE clause patterns — range conditions (`BETWEEN`, `>`, `<`), multi-condition ANDs/ORs, and LIKE patterns. If your 15K dataset was generated from templates, deliberately expand the template variety for WHERE clauses.

2. Add hard negatives: for each training example with a correct WHERE clause, generate a synthetic near-miss example where the WHERE clause is subtly wrong (wrong operator, off-by-one threshold) and mark it as incorrect. This teaches the model to discriminate fine-grained filter conditions.

3. Augment with schema-specific training: if your eval schemas contain columns like `created_at`, `status`, or `amount` that require domain-specific filters (`WHERE status = 'active'`, `WHERE created_at > NOW() - INTERVAL '30 days'`), add more training examples that cover exactly those column-filter combinations.

---

## Q9

**Model answer:** Three concrete next steps:

First, run full error analysis on the 33 failing examples (67% correctness = 67 correct, 33 wrong out of 100). Categorize by failure type. If wrong-WHERE or wrong-aggregation dominates, add 2,000–3,000 targeted synthetic examples covering exactly those patterns and run another QLoRA fine-tune for 3 epochs. This alone should push execution correctness to 72–78%.

Second, improve the training data quality signal. Your current 15K dataset was generated synthetically (Week 37). Add a self-consistency filter: generate 5 SQL candidates per question using temperature sampling, execute all 5, and only keep examples where at least 3 agree on the result set. This filters out noisy training examples that teach the model contradictory mappings.

Third, plan the Phase 5 GRPO run. Supervised fine-tuning is limited because the training signal is binary (correct SQL / incorrect SQL text). GRPO uses the execution harness you built this week as a reward signal — the model generates multiple SQL candidates per question, each is executed, and reward is proportional to correctness. This is the intervention that typically pushes text-to-SQL models from the 70–75% plateau to 85%+ because it directly optimizes for the metric you care about rather than token-level imitation of training examples.
