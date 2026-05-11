# Week 54 Quiz — Synthetic Data Generation

## Multiple Choice

**Q1.** Magpie-style data generation differs from classic Self-Instruct in which key way?

A) Magpie uses a separate discriminator model to filter low-quality examples.
B) Magpie feeds only the system prompt + partial user turn, letting the model generate both the instruction and the response self-consistently.
C) Magpie requires human annotation of every generated example before use.
D) Magpie generates data from a predefined template library rather than open-ended prompting.

---

**Q2.** You generate 10,000 SQL examples and measure a 42% overall execution rate. Which is the most likely root cause?

A) Your PostgreSQL version is too old to support the generated syntax.
B) Your teacher model has insufficient SQL knowledge.
C) Your prompts are not grounding the teacher in the target schema, causing hallucinated column names.
D) The teacher model temperature (0.8) is too high.

---

**Q3.** You are generating 30K examples with 10 concurrent async API calls. After 2,000 examples your pipeline starts receiving HTTP 429 (rate limit) errors every 30 seconds. What is the correct fix?

A) Switch to a synchronous (sequential) pipeline to avoid rate limits.
B) Implement a semaphore to limit concurrency and add exponential backoff on 429 responses.
C) Increase the number of concurrent calls to 50 to complete faster before the rate limit resets.
D) Cache all API responses locally and replay them to avoid further API calls.

---

**Q4.** Genie-style grounded generation includes the target schema DDL in every prompt. What problem does this primarily solve?

A) It prevents the teacher from generating overly long queries.
B) It eliminates hallucinated table and column names that reference schema elements not present in the target database.
C) It forces the teacher to use PostgreSQL dialect instead of MySQL.
D) It allows the pipeline to skip execution validation.

---

## Short Answer

**Q5.** Your generation pipeline produces 1,000 examples for the "window functions" skill. Manual inspection shows that 80% of them use only `ROW_NUMBER()` and `RANK()`, while `LAG`, `LEAD`, `NTILE`, `FIRST_VALUE`, `LAST_VALUE`, and `NTH_VALUE` are almost absent. What is the likely cause and how do you fix it?

---

**Q6.** You plan to generate 15K examples using GPT-4o (~$5/M input tokens) and 15K using GPT-4o-mini (~$0.15/M input tokens). Design a principled allocation: which skill categories and difficulty levels should each model handle? Justify your choice.

---

**Q7.** After generating 30K raw examples, your execution rate for TimescaleDB-specific queries is 38%, while standard PostgreSQL queries achieve 71%. You have 2 weeks before training starts. What are your three options (ranked by effectiveness)?

---

## Deep Scenario

**Q8.** You have finished generating 30K raw SQL pairs. Before filtering (Week 55), you run a quick analysis and find:
- 8,200 exact or near-duplicate pairs
- 11,300 with execution_status = "pass"
- 8,100 with execution_status = "fail" (of which 3,200 fail due to missing extension — TimescaleDB not loaded)
- 2,400 with parse failures (malformed JSON from teacher)

After deduplication and fixing the extension issue and re-running, what is your realistic usable dataset size? Show your calculation. Is this sufficient for your v3 target of 50K? If not, what do you do next?
