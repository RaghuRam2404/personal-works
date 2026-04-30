# Week 54 Answers

## Q1 — Answer: B

**Why:** The Magpie insight is that a model fine-tuned on chat data has learned to generate appropriate user instructions given a system prompt context. By stopping the input at the beginning of the user turn, the model completes the instruction naturally, then generates the response. This produces self-consistent instruction-response pairs with the exact stylistic distribution of that model family — making the data on-policy for fine-tuning.

**Why A is wrong:** Magpie uses no discriminator; filtering comes from external execution or quality checks.

**Why C is wrong:** Magpie's advantage is precisely that it eliminates the human annotation bottleneck.

**Why D is wrong:** Magpie's output is open-ended and instruction-driven, not template-based.

---

## Q2 — Answer: C

**Why:** A 42% execution rate is the classic symptom of schema hallucination. When the teacher model doesn't see the actual DDL, it invents plausible-sounding column names based on general priors. These columns don't exist in your target schema, causing "column does not exist" errors at execution time. The fix is Genie-style grounding: include the full DDL in every prompt.

**Why A is wrong:** PostgreSQL version issues would produce a specific error pattern (function not found, syntax not supported) not 58% failure.

**Why B is wrong:** GPT-4o and Claude have strong SQL knowledge. The failure is about schema knowledge, not SQL knowledge.

**Why D is wrong:** Temperature of 0.8 increases diversity but doesn't cause a systematic 58% failure rate.

---

## Q3 — Answer: B

**Why:** The correct solution is to reduce concurrency (via asyncio.Semaphore) and handle 429 errors with exponential backoff (wait 2^n seconds before retrying). This keeps the pipeline running without manual intervention.

**Why A is wrong:** Sequential requests are far too slow for 30K examples and don't prevent rate limits — they just hit them more slowly.

**Why C is wrong:** Increasing concurrency worsens the rate limiting problem.

**Why D is wrong:** Caching is useful for identical requests (repeated prompts), but each generation call has different prompts.

---

## Q4 — Answer: B

**Why:** Without schema grounding, the teacher model generates SQL using column names drawn from its pretraining distribution — which includes many standard schema patterns (users, orders, products, etc.) but not your specific PostgreSQL/TimescaleDB schemas (sensor_readings, metric_data, events). Schema hallucination is the #1 failure mode for SQL generation and the primary motivation for the Genie approach.

---

## Q5 — Model Answer

The root cause is frequency bias in the teacher's pretraining data. `ROW_NUMBER()` and `RANK()` appear far more frequently in online SQL content than the other window functions, so the teacher defaults to them when the prompt doesn't explicitly request otherwise.

Fix: add explicit window function enumeration to your prompt. Change "Generate queries using window functions" to "Generate queries using each of these window functions at least once: LAG, LEAD, NTILE, FIRST_VALUE, LAST_VALUE, NTH_VALUE. Do NOT use ROW_NUMBER or RANK." Then generate separate batches for each function to ensure balanced coverage.

---

## Q6 — Model Answer

Allocation principle: use expensive models (GPT-4o) only where cheaper models fail, and route by difficulty × skill rarity.

GPT-4o (15K examples): all Expert-difficulty examples, all TimescaleDB-specific examples (hyperfunctions, continuous aggregates, compression), multi-table CTEs, and any skill where GPT-4o-mini's execution rate is below 50% in a 100-example pilot.

GPT-4o-mini (15K examples): Easy and Medium-difficulty standard PostgreSQL (basic joins, aggregations, filters, ORDER BY, simple GROUP BY, single-table window functions). These are well-represented in the teacher's pretraining data and mini-class models handle them reliably.

Justification: GPT-4o costs ~33× more than mini. Spending the premium on hard/rare skills, where quality matters most and mini fails most, maximizes ROI. Easy examples don't need frontier intelligence — they need volume and diversity.

---

## Q7 — Model Answer (ranked)

1. **Fix the test execution environment.** Re-run the 3,200 failures that failed only due to missing TimescaleDB extension (install the extension in your test DB and re-validate). This is free and recovers ~4 percentage points of execution rate immediately.

2. **Revise prompts with more TimescaleDB few-shot examples.** Add 3–4 verified working TimescaleDB queries to each TimescaleDB-skill prompt. Re-generate the lowest-performing 20% of TimescaleDB examples with the improved prompts.

3. **Hand-write 200 verified TimescaleDB examples** and use them as the core of a smaller, higher-quality TimescaleDB split. Supplement with teacher generation using these 200 as few-shot context. 200 verified + 800 teacher-generated with verified few-shot will outperform 2,000 unconstrained teacher examples.

---

## Q8 — Calculation

Starting with 30,000 raw examples:
- Remove 2,400 parse failures → 27,600 remain
- Remove 8,200 near-duplicates → 19,400 remain
- Of the 19,400: 11,300 "pass" + 4,900 "fail" (8,100 - 3,200 fixed by extension fix) = 11,300 + 3,200 re-validated = ~12,500 pass after extension fix + ~1,700 genuinely bad SQL fail
- Realistic usable: ~12,500–13,500 examples

This is well short of 50K. The v2 dataset (18K after dedup from Phase 4–5) plus 13K from generation = ~31K. Still short.

Next steps:
1. Augment Spider/BIRD training examples (rewritten for PostgreSQL dialect) — can add 8–10K
2. Run a second generation pass targeting specifically the failed skills with improved prompts — expect 60% execution rate → ~4K more
3. Accept 40–45K as the v3 target if week budget is exhausted; quality at 40K well-curated > quantity at 50K poorly curated (LIMA principle from Week 53)
