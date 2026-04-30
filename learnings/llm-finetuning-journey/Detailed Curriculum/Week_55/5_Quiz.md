# Week 55 Quiz — LLM-as-Judge and Aggressive Filtering

## Multiple Choice

**Q1.** The Alpagasus paper filters a 52K instruction dataset to 9K examples using GPT-4 as judge. The resulting model outperforms the model trained on all 52K examples. Which explanation is most accurate?

A) GPT-4 as judge introduces knowledge distillation that improves the student beyond simple filtering.
B) The 43K removed examples contain contradictory or misleading training signal that degrades the full dataset model.
C) The 9K examples overfit the model, causing memorization rather than generalization, which happens to score higher on benchmarks.
D) The judge score correlates with training example token length, and shorter examples train faster.

---

**Q2.** You set your judge temperature to 0.8 when scoring examples. A colleague says you should use temperature=0.0. Who is right and why?

A) 0.8 is correct — randomness ensures diverse judgments that average out to a reliable score.
B) 0.0 is correct — judges should be deterministic; the same example should always receive the same score.
C) Both are wrong — judge temperature should match the training temperature (typically 0.7).
D) It does not matter for small datasets (< 30K); temperature only affects quality at > 1M examples.

---

**Q3.** After filtering your 30K raw dataset, your TimescaleDB skill has only 80 examples remaining. You apply score ≥ 3 relaxation and recover 120 more. You now have 200 TimescaleDB examples. Is this enough to train a reliable TimescaleDB skill?

A) Yes — 200 examples is sufficient for any SQL construct given a strong pretrained model.
B) No — 200 examples will teach the model the syntax but not reliable generalization across diverse schemas.
C) Yes — but only if you use a high learning rate to emphasize these rare examples.
D) No — you need at least 10,000 examples of any skill for fine-tuning to have measurable effect.

---

**Q4.** Your calibration set shows judge agreement of 72% and Cohen's kappa of 0.51. Which change is most likely to improve both metrics?

A) Increase the calibration set size from 100 to 500 examples.
B) Add explicit few-shot examples (one per score level) with reasoning to the judge prompt.
C) Switch from GPT-4o-mini to GPT-4o for judging.
D) Change the scale from 1-5 to binary (keep/reject) to reduce ambiguity.

---

## Short Answer

**Q5.** Your LLM judge consistently rates verbose SQL (with many aliases, CTEs, and comments) higher than functionally equivalent concise SQL. This bias is well-documented in the literature. Propose two concrete rubric modifications that specifically counter this bias for your SQL use case.

---

**Q6.** You are filtering 15,000 execution-passing examples using an LLM judge at $0.30/1M tokens (GPT-4o-mini). Your judge prompt is 200 tokens, the SQL is ~150 tokens, and the output is ~50 tokens. Estimate the total filtering cost. Is it worth it?

---

**Q7.** After filtering, your dataset has 22,000 examples. You need 50,000 for your original target. You can either: (a) relax all score thresholds to score ≥ 3, recovering ~8,000 more examples, or (b) run another generation pass targeting specific skill gaps to add ~20,000 new raw examples (which after filtering might yield 12,000–14,000). Compare the tradeoffs.

---

## Deep Scenario

**Q8.** You discover after filtering that your dataset has an unintended bias: 65% of examples have schemas with only 2–3 tables, because your Week 54 generation prompts defaulted to simple schemas. Your actual target use case (TimescaleDB dashboards) commonly involves 8–12 table joins. Your training starts in 3 weeks. Design a remediation plan with specific actions, timeline, and expected impact on the final model.
