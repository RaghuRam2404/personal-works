# Week 56 Answers

## Q1 — Answer: C

**Why:** In a multi-turn conversation training setup, you compute loss only on the assistant turns (all of them, not just the final one). The model's job is to predict SQL responses; computing loss on user question tokens would train the model to predict user behavior, which is wrong. Computing loss on all assistant turns — not just the last — gives you more gradient signal per example and makes training more efficient. The system prompt and user turns are masked to zero loss contribution.

**Why B is wrong:** Training on only the final turn wastes the signal from earlier assistant turns.

**Why A is wrong:** Training on user tokens corrupts the gradient and causes poor results.

---

## Q2 — Answer: A

**Why:** PostgreSQL does not support `strftime()` — that is a SQLite/Python function. The PostgreSQL equivalents for extracting year are `EXTRACT(year FROM col)` (ANSI SQL, preferred) or `date_part('year', col)` (PostgreSQL-specific). `TO_CHAR(col, 'YYYY')` also works but returns text, not numeric. sqlglot handles most strftime patterns but date extraction is a known imperfect conversion that requires manual verification.

---

## Q3 — Answer: B or D (both valid; B is faster, D is more principled)

**Why B:** Keeping only the last 3 turns preserves the most recent context (which is what the model needs to predict the final response) and fits within the 4,096 token limit. The trade-off: the model won't see the full conversation history, but for 6-turn conversations the first few turns are often scene-setting that matters less for the final prediction.

**Why D:** Splitting into two 3-turn conversations preserves all the conversational content as separate training examples. The cost: the split conversations may lose coherence if they refer to results established in turns they no longer have access to.

**Why C is dangerous:** Automatic truncation in the middle of a conversation can cut off the beginning of a SQL query, producing malformed input.

---

## Q4 — Answer: A

**Why:** The model has lost track of which tables were established in the conversational context. This is context hallucination — the model generates plausible SQL for the user's question but doesn't restrict itself to the schema state established by the prior turns. This is exactly why training on multi-turn data with proper context masking matters: the model must learn to read the full conversation history to determine which tables are "in scope."

---

## Q5 — Model Answer

Multi-turn fraction: 4,500 / 27,000 ≈ 16.7%.

This ratio is reasonable but on the low end. For an application where multi-turn SQL is a primary use case (analytics dashboards, iterative data exploration), 20–30% multi-turn is a stronger target. At 16.7%, the model will handle basic multi-turn cases but may struggle with longer (4+ turn) conversations or highly implicit references.

If multi-turn is a key product feature, aim for 25% minimum. The additional cost is modest: generating 2,000 more multi-turn conversations takes roughly the same compute as 8,000 single-turn examples, but each conversation is worth 3× in terms of contextual training signal.

---

## Q6 — Model Answer

The failure occurs because "Which products are above average?" contains an implicit reference: "above average" refers to the average of the totals computed in turn 1. A model trained only on single-turn examples has never seen a question where "average" refers to a value computed in a prior query — it always computes the average from scratch.

The model will likely generate a query that computes average sales across all products and all time (ignoring the Q4 2024 constraint) or asks for the average of a specific column without the self-referencing structure required:

```sql
-- What the model would likely generate (wrong):
SELECT product, total_sales
FROM (SELECT product, SUM(amount) as total_sales FROM orders GROUP BY product) t
WHERE total_sales > (SELECT AVG(amount) FROM orders);  -- wrong: avg of raw amounts, not totals

-- What the correct answer requires:
WITH q4 AS (
    SELECT product, SUM(amount) as total_sales
    FROM orders
    WHERE date >= '2024-10-01' AND date < '2025-01-01'
    GROUP BY product
)
SELECT product, total_sales
FROM q4
WHERE total_sales > (SELECT AVG(total_sales) FROM q4);
```

The multi-turn training teaches the model that the CTE from turn 1 is the reference point for turn 2's "average."

---

## Q7 — Model Answer

Coherence verification method:

1. Extract all table/column names and date references from turn N's SQL.
2. Extract all noun phrases from turn N+1's user question using simple NLP (spaCy or keyword matching).
3. Compute overlap: does turn N+1's question contain at least one entity (table alias, column name, or entity from turn N's result schema) that matches turn N's SQL?
4. If overlap is zero: the conversation is likely incoherent. Flag for manual review.
5. Additionally, verify that turn N+1's SQL uses at least one table also used in turn N (for filter/aggregation refinements) OR adds a new table that is joined to turn N's core table (for join additions).

This is imperfect (keyword overlap can miss semantic references like "those sensors" → sensors table) but catches the worst cases: completely unrelated turn 2 questions. Supplement with a 5% manual review of flagged conversations.

---

## Q8 — Two Example Conversations

**Example 1 (4-turn, implicit reference):**

Schema: `sensor_readings(sensor_id, time, temperature, humidity)`, `sensors(sensor_id, name, building, floor)`

Turn 1: "Show average temperature per floor for yesterday."
```sql
SELECT s.floor, AVG(r.temperature) as avg_temp
FROM sensor_readings r JOIN sensors s ON r.sensor_id = s.sensor_id
WHERE r.time::date = CURRENT_DATE - 1
GROUP BY s.floor ORDER BY s.floor;
```

Turn 2: "Which floors are above 28 degrees?"
```sql
SELECT s.floor, AVG(r.temperature) as avg_temp
FROM sensor_readings r JOIN sensors s ON r.sensor_id = s.sensor_id
WHERE r.time::date = CURRENT_DATE - 1
GROUP BY s.floor HAVING AVG(r.temperature) > 28 ORDER BY avg_temp DESC;
```

Turn 3: "Show those same floors' hourly trend for yesterday."
```sql
SELECT s.floor, time_bucket('1 hour', r.time) as hour, AVG(r.temperature) as avg_temp
FROM sensor_readings r JOIN sensors s ON r.sensor_id = s.sensor_id
WHERE r.time::date = CURRENT_DATE - 1
  AND s.floor IN (SELECT s2.floor FROM sensor_readings r2 JOIN sensors s2 ON r2.sensor_id=s2.sensor_id
                  WHERE r2.time::date = CURRENT_DATE - 1 GROUP BY s2.floor HAVING AVG(r2.temperature)>28)
GROUP BY s.floor, hour ORDER BY s.floor, hour;
```

Turn 4: "Add gap-filling for any missing hours."
```sql
SELECT s.floor, time_bucket_gapfill('1 hour', r.time) as hour,
       LOCF(AVG(r.temperature)) as avg_temp
FROM sensor_readings r JOIN sensors s ON r.sensor_id = s.sensor_id
WHERE r.time >= CURRENT_DATE - 1 AND r.time < CURRENT_DATE
  AND s.floor IN (...)  -- same subquery as above
GROUP BY s.floor, hour ORDER BY s.floor, hour;
```
