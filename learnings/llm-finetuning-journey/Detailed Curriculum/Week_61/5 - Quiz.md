# Week 61 Quiz — Evaluation Harness

## Multiple Choice

**Q1.** You compare your model to GPT-4o using execution accuracy on BIRD-SQL dev set. Your model scores 63%, GPT-4o scores 72%. You claim "our model achieves 87.5% of GPT-4o's performance." What is a more scientifically honest way to present this result?

A) Report the absolute gap: "Our model lags GPT-4o by 9 percentage points on BIRD-SQL dev."
B) Report in terms of relative error: "Our model's error rate is (37%)/(28%) = 1.32× GPT-4o's error rate."
C) Report on your custom domain benchmark where your model may be closer to or exceed GPT-4o.
D) All of the above are valid; present all three framings to give a complete picture.

---

**Q2.** You run eval on 200 examples and get 68% execution accuracy. You want to report a 95% confidence interval. Using bootstrapping (resample 1,000 times), you get CI = [63%, 73%]. What does this mean for comparing to a published paper reporting 70% on the same benchmark without confidence intervals?

A) Your model is worse than the paper's model — 68% < 70%.
B) The results are statistically indistinguishable — the CI overlaps 70%.
C) Your model is better — your confidence interval shows you could be 73%.
D) You cannot compare because your bootstrap method is different from the paper's.

---

**Q3.** Your model generates `SELECT * FROM sensor_readings ORDER BY time DESC LIMIT 100;` but the reference answer is `SELECT * FROM sensor_readings WHERE time > NOW() - INTERVAL '24 hours' ORDER BY time DESC;`. Both execute without error. For a question asking "show the 100 most recent sensor readings from the last 24 hours," which execution accuracy outcome is correct?

A) Correct — both queries return sensor readings in time-descending order.
B) Incorrect — the result sets are different (LIMIT 100 may include readings older than 24 hours).
C) Correct — execution accuracy only checks that the query runs, not that it returns the right rows.
D) Cannot determine without knowing the actual data in the database.

---

**Q4.** The Defog sql-eval benchmark uses "real-world" enterprise SQL questions. Your model scores 71% on Defog but only 62% on BIRD-SQL. What is the most likely explanation?

A) Defog is easier than BIRD — it uses simpler schemas and queries.
B) Your training data distribution is more similar to Defog's enterprise SQL style than to BIRD's academic queries.
C) Defog uses GPT-4 as a judge while BIRD uses exact match, inflating Defog scores.
D) BIRD has a stricter execution comparison that penalizes equivalent queries.

---

## Short Answer

**Q5.** Explain why using the test set for model selection (choosing between SFT-v3, DPO-v3, and GRPO-final based on test set performance) is methodologically wrong, even if the test set labels are public.

---

**Q6.** Your model correctly generates a SQL query that returns the right answer for 70% of BIRD-SQL questions. However, for 15% of questions, the model generates valid SQL that produces correct-looking results but is semantically wrong for subtly different reasons than your reference answer. How does execution accuracy over/underestimate your model's true capability?

---

## Deep Scenario

**Q7.** Your BIRD-SQL evaluation shows:
- Simple questions: 78% accuracy
- Moderate questions: 64% accuracy  
- Challenging questions: 41% accuracy

The challenging questions include: multi-table CTEs with window functions, queries requiring external knowledge, and questions with implicit schema relationships. Diagnose what's driving the drop in each sub-category and propose targeted improvements to your training data for each.
