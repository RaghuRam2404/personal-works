# Week 56 Quiz — Conversational Multi-Turn SQL

## Multiple Choice

**Q1.** When training on multi-turn conversations with a decoder-only model, which tokens should contribute to the training loss?

A) All tokens in the conversation — the model must predict the full context.
B) Only the final assistant turn — the model only predicts the last response.
C) Only assistant turns — the model must predict all SQL responses but not user questions.
D) Only user turns in later conversation positions — these represent the hardest predictions.

---

**Q2.** You convert a CoSQL example from SQLite to PostgreSQL using sqlglot. The original SQLite uses `strftime('%Y', date_column)`. After conversion, the query fails in Postgres. What is the most likely fix?

A) Replace with `EXTRACT(year FROM date_column)` or `date_part('year', date_column)`.
B) Add `CAST(date_column AS TEXT)` before the strftime call.
C) The query cannot be converted — CoSQL examples with date functions are incompatible with PostgreSQL.
D) Upgrade your sqlglot version; strftime conversion is supported in sqlglot >= 20.0.

---

**Q3.** Your 6-turn TimescaleDB conversation has 4,800 tokens. Your model's maximum context length is 4,096. What should you do?

A) Skip this conversation — it exceeds the context limit.
B) Keep only the last 3 turns of the conversation.
C) Use the full 4,800 tokens; the model will truncate automatically during training.
D) Split it into two separate conversations of 3 turns each.

---

**Q4.** A user asks in turn 2: "Now show only the ones from building A." Your model responds with a query that filters on `building = 'A'` but joins a completely different table not present in turn 1's query. What failure mode does this illustrate?

A) Context hallucination — the model lost track of which tables were established in turn 1.
B) Schema overfitting — the model is using a table it memorized from training data.
C) Length generalization failure — the model can't handle 2-turn contexts.
D) Temperature instability — the model is sampling from a different mode.

---

## Short Answer

**Q5.** You have 2,500 valid CoSQL examples (converted to PostgreSQL) and 2,000 synthetic TimescaleDB multi-turn examples. Your total v3 dataset after merging is 27,000 examples (23,500 single-turn + 4,500 multi-turn). Roughly what fraction of your training examples is multi-turn, and is this ratio appropriate? Justify your answer.

---

**Q6.** Explain why a model trained only on single-turn SQL is likely to fail the following 2-turn interaction, even though it has seen both queries individually in training:

Turn 1: "Show total sales by product for Q4 2024."
Turn 2: "Which products are above average?"

---

**Q7.** Describe a systematic method to verify that your synthetic multi-turn conversations are coherent — that is, each turn genuinely references or refines the previous turn's query rather than asking an unrelated question.

---

## Deep Scenario

**Q8.** You are designing an evaluation set specifically for multi-turn SQL capability. You want to measure how well your model tracks context across turns. Design a 10-example evaluation set for the TimescaleDB domain covering:
- 3 different types of refinement (filter, aggregation change, join addition)
- At least 2 examples where the user uses a pronoun or implicit reference ("those sensors," "same period")
- At least 2 examples with 4+ turns

Write 2 complete example conversations (with question and gold SQL for each turn) to illustrate what your eval set looks like.
