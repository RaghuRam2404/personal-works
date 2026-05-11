# Week 55 Answers

## Q1 — Answer: B

**Why:** The 43K removed examples are not neutral — they contain low-quality, contradictory, or misleading instruction-response pairs. When the model trains on all 52K examples, the gradient signal from the bad 43K actively fights against the signal from the good 9K. Removing the bad data doesn't just shrink the dataset; it removes confusing gradient directions and allows the model to learn the correct signal more cleanly. This is the core lesson of data-centric AI.

**Why A is wrong:** There is no distillation mechanism in simple filtering. The judge only selects; it does not inject knowledge.

**Why C is wrong:** The filtered model generalizes better, not worse. Overfitting would manifest as improved training metrics but degraded generalization — the opposite of what is observed.

---

## Q2 — Answer: B

**Why:** A judge is a classifier, not a generator. You want consistent, reproducible decisions. If the same example receives score 4 on one judge call and score 2 on another due to temperature randomness, your filtering is non-deterministic and cannot be audited or reproduced. Always use temperature=0.0 (or the API equivalent) for judging and scoring tasks.

---

## Q3 — Answer: B

**Why:** 200 examples is typically below the threshold for reliable skill generalization in fine-tuning. The model will learn the pattern for the specific schemas and question phrasings in those 200 examples but will likely fail on novel schemas or paraphrased questions. Research suggests 500–1,000 examples per skill for reliable generalization; 2,000+ for near-ceiling performance. At 200 examples, you should plan for additional generation in a subsequent iteration (Week 75) and flag TimescaleDB as a known limitation in your technical report.

**Why A is wrong:** The "strong pretrained model" assumption only helps if the skill is well-represented in pretraining data. TimescaleDB is niche enough that pretraining provides minimal help.

**Why D is wrong:** The 10,000-example rule is for pretraining; fine-tuning requires far fewer examples because the pretrained model already has relevant knowledge.

---

## Q4 — Answer: B

**Why:** Adding few-shot examples with explicit reasoning at each score level is the most direct fix for judge miscalibration. The judge sees concrete examples of what a 1 looks like, what a 3 looks like, etc., with the reasoning explained. This reduces ambiguity in the rubric's application. Agreement of 72% at kappa 0.51 indicates the rubric is unclear at boundary cases (2 vs 3, 3 vs 4) — examples address exactly these boundaries.

**Why A is wrong:** More calibration data reveals the disagreement pattern but doesn't fix the underlying prompt ambiguity.

**Why C is wrong:** GPT-4o might improve slightly but the root cause is prompt design, not model capability.

---

## Q5 — Model Answer

Two concrete rubric modifications:

1. "Brevity bonus, within correctness: among two SQL answers that are equally correct, prefer the shorter one. If a CTE could be eliminated by a simpler subquery without loss of clarity, the CTE version scores one point lower. Do not award style points for additional boilerplate."

2. "Explicitly penalize unnecessary verbosity: aliases that add no clarity (e.g., `t1`, `col_a` instead of descriptive names) are neutral, but structures that add length without correctness benefit (redundant GROUP BY columns, unused CTEs, comments that restate the obvious) reduce the score by 1 point."

---

## Q6 — Cost Calculation

Per example: 200 (prompt) + 150 (SQL) + 50 (output) = 400 tokens. At $0.30/1M tokens = $0.00030/1K tokens = $0.00012 per example.

15,000 examples × $0.00012 = $1.80 total.

Yes, it is absolutely worth it. Less than $2 to filter 15,000 examples — this is the highest-ROI spend in your entire project. The filtering typically yields a 10–20 percentage point improvement in model evaluation scores, which no amount of additional training compute can replicate.

---

## Q7 — Model Answer

Option (a) — relax threshold to score ≥ 3: Fast (no generation needed), recovers 8,000 examples immediately. Trade-off: deliberately adds examples you previously deemed borderline — the judge saw real problems in them. These examples add noise. Useful only if the relaxation is targeted (rare skills only, not across the board).

Option (b) — new generation pass: Takes 1–2 weeks, costs additional API fees (~$15–20), and yields 12,000–14,000 high-quality examples after filtering. Superior choice because it adds genuinely new examples rather than reviving rejected ones, and you can target the specific schema diversity gap (8–12 table joins).

Recommendation: do a limited option (a) for skills below 150 examples (to ensure minimum coverage), then do option (b) targeted at the specific schema diversity gap. The combination costs roughly 1 extra week and $20 but produces a meaningfully stronger dataset.

---

## Q8 — Model Answer

Diagnosis: the schema diversity bias (65% 2–3 table schemas vs. target of 8–12) will cause the model to underperform on complex join queries, which are exactly the production queries in TimescaleDB dashboards.

Remediation plan (3 weeks before training starts):

Week 1 (this week): Generate 3K new examples using 8–12 table schemas exclusively. Create 5 complex TimescaleDB schemas (IoT metrics + devices + locations + thresholds + alerts tables) and use them in all new generation prompts. Target skill: multi-table joins + TimescaleDB hyperfunctions in the same query. Filter immediately with judge (temperature=0.0). Expect ~60% pass rate → ~1,800 good examples.

Week 2 (immediately after): Augment Spider training split — Spider has multi-table schemas (up to 20 tables). Re-purpose Spider training questions that use 4+ tables, convert SQL to PostgreSQL dialect via sqlglot, and add to your dataset. This adds ~2,000 verified examples with complex schemas at zero generation cost.

Expected impact: shifting from 65% 2–3 table → 45% 2–3 table / 35% 4–7 table / 20% 8–12 table. This more closely mirrors the production schema complexity distribution. Model accuracy on complex join queries should improve 15–25% relative on your custom benchmark.

If time runs out: document the bias explicitly in your technical report as a known limitation and plan remediation in Week 75 (iteration sprint).
