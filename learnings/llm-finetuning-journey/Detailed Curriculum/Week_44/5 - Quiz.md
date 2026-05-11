# Week 44 Quiz — Building Preference Data for SQL

## Multiple Choice

**Q1.** Execution-based preference labeling for SQL is categorized as which type of feedback signal?

A) RLHF — Reinforcement Learning from Human Feedback  
B) RLAIF — Reinforcement Learning from AI Feedback  
C) Rule-based reward — verifiable reward from a deterministic process  
D) Constitutional AI — reward derived from a set of written principles  

---

**Q2.** You have 3000 candidate pairs from your execution harness. 900 pairs have both SQL_A and SQL_B executing correctly with the same row count. What should you do with these pairs for DPO training?

A) Include them with a random choice of which is "chosen" — equal quality means equal probability  
B) Include them with SQL_A always as "chosen" since it comes from your superior SFT model  
C) Discard them — DPO requires a clear quality signal between chosen and rejected  
D) Include them with a 0.5 label weight in a soft DPO variant  

---

**Q3.** When labeling SQL preferences by execution, you find that 70% of pairs are discarded (both fail). The most likely cause is:

A) The Postgres database is too slow to execute queries within the timeout  
B) The prompt distribution does not match the training schema, causing widespread hallucination  
C) The execution harness has a bug causing all queries to return success=False  
D) DPO is not compatible with execution-based labels  

---

**Q4.** Constitutional AI (CAI) is most useful for SQL preference labeling in which scenario?

A) When you have no Postgres database to execute queries against  
B) When you want to add execution-based rewards to your GRPO training  
C) When you need ground-truth row counts for verification  
D) When labeling is done entirely by human annotators  

---

## Short Answer

**Q5.** You have a preference pair where both SQL_A and SQL_B execute successfully but return different row counts (A returns 15 rows, B returns 12 rows). The expected output is 12 rows. Describe the correct labeling decision and what additional infrastructure you need to make this determination programmatically at scale.

---

**Q6.** Your labeling pipeline produces 2500 pairs. 1800 have "v1 model chosen", 700 have "base model chosen". From a DPO training perspective, is this imbalance a problem? How would you address it, if at all?

---

**Q7.** Explain the difference between the preference data you are building this week (used for DPO in Week 45) and the reward signal you will design for GRPO in Weeks 47–48. In one sentence each: what is the data format, when is the signal computed, and what is it based on?

---

## Deep Scenario

**Q8.** You finish building your preference dataset and push it to HuggingFace Hub. A colleague reviews it and raises two concerns:

Concern 1: "The 'chosen' SQL always executes, but 30% of them return incorrect results — they query the right table but use the wrong WHERE clause. They just happen to return some rows, not the right rows."

Concern 2: "The 'rejected' SQL has syntax errors 80% of the time. This means DPO will mostly learn to avoid syntax errors, not to generate correct SQL."

Evaluate both concerns. Are they valid? If so, propose a concrete fix for each.
