# Week 69 Quiz — Evaluation and Ablations

## Multiple Choice

**Q1.** Your paper reports exact-match (EM) accuracy of 83.1% on your Custom-200 benchmark. A reviewer points out that EM is "too strict" and requests execution accuracy (EX) instead. What is the key tradeoff that makes EM simpler but less valid than EX?

A. EM requires more compute because it needs a PostgreSQL instance
B. EX is always higher than EM because it accepts more SQL variants; EM penalizes correct SQL that differs in syntax
C. EM is harder to implement than EX
D. EX does not work for TimescaleDB-specific queries

**Q2.** You present an ablation table showing that adding GRPO to DPO improved Custom-200 accuracy by 0.5 pp (82.6% → 83.1%). A reviewer says: "This improvement is not statistically significant on a 200-example benchmark." What is the correct response?

A. Remove the GRPO ablation from the paper since the improvement is not significant
B. Acknowledge that 200 examples gives high variance; compute a 95% confidence interval (approximately ±3.1 pp for a 200-example binary test) and note that 0.5 pp is within that bound
C. Run the evaluation 5 times and report the mean to reduce variance
D. Switch to a larger benchmark for the ablation only

**Q3.** You discover that 35% of your model's failures on Custom-200 involve missing `time_bucket` usage. What is the most direct remediation for future model versions?

A. Switch from GRPO to PPO — PPO handles time-series patterns better
B. Increase the DPO β to reduce KL penalty from the reference model
C. Add more `time_bucket` training examples and re-run the SFT stage with the augmented dataset
D. Use a larger base model (13B instead of 7B)

**Q4.** You want to add a competitor model (SQLCoder-7B) to your results table but you did not run it on your Custom-200 benchmark. You found their published Spider 1.0 result (80.2%). How should you handle this in the table?

A. Run SQLCoder-7B on Custom-200 yourself, since published results on different benchmarks are not comparable
B. Report their Spider result in the Spider column with a † mark; leave Custom-200 as — since you did not run it
C. Interpolate their Custom-200 result from their Spider performance using a regression model
D. Email the Defog team and ask for their Custom-200 numbers

## Short Answer

**Q5.** Define exact-match normalization for SQL. List three normalization steps that are universally applied before comparing generated SQL to reference SQL.

**Q6.** Your ablation study shows that CPT provided a 6.3 pp gain on Custom-200. A colleague argues: "Maybe the 6.3 pp gain came from the additional GPU time during CPT, not from the domain-specific content." How would you design an experiment to test whether CPT content vs CPT compute time drives the gain?

**Q7.** You report that your model outperforms GPT-4o on Custom-200 (83.1% vs 79.4%). A press release from your team claims "open-source model beats GPT-4o at SQL." What is the most important caveat that must accompany this claim?

## Deep Scenario

**Q8.** Three months after publishing your technical report, another team fine-tunes the same base model (Qwen2.5-Coder-7B) on a similar PostgreSQL dataset and reports 87.3% on your Custom-200 benchmark — 4.2 pp better than your published result.

Write a 150–200 word analysis that: (a) identifies two possible sources of the gap (their pipeline was genuinely better vs other explanations), (b) describes one experiment you could run within one week to determine whether the gap is real or methodological, and (c) evaluates whether this result invalidates your technical report's contribution.
