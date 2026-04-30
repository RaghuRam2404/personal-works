# Week 50 Quiz — Iteration and Debugging

## Multiple Choice

**Q1.** You are running a targeted GRPO iteration on v3. You changed two things simultaneously: (1) expanded the dataset with complex prompts and (2) reduced β from 0.05 to 0.02. v3-iter1 is 8pp better on complex queries and KL divergence also increased from 2 to 9 nats. What is the problem with this experimental design?

A) Reducing β is always wrong — it should never be changed during iteration  
B) You changed two variables simultaneously and cannot determine which change drove the improvement or the KL increase  
C) The dataset expansion is invalid if the prompts were not verified for 20–60% success rate  
D) GRPO iteration should always start from v1, not from v3  

---

**Q2.** Your iteration hypothesis is: "v3 fails on TimescaleDB time_bucket queries because the preference dataset had no time_bucket examples." The correct experiment to test this is:

A) Re-run the full Week 48 GRPO training from v2 with 500 time_bucket prompts added  
B) Add 100 time_bucket prompts to the GRPO training set and run 300 steps from v3  
C) Re-do the Week 44 preference dataset construction with time_bucket prompts and re-run DPO  
D) Add time_bucket prompts to the eval set and see if v3 already handles them  

---

**Q3.** After your iteration run, v3-iter1 has higher execution accuracy (+5pp) but the W&B run shows `reward_std` was near 0 for the last 200 of 300 steps. What does this imply about the quality of the improvement?

A) The improvement is solid — low reward_std means the model converged properly  
B) The improvement came mostly from the first 100 steps; the last 200 steps produced near-zero gradient and may have degraded quality slightly  
C) The model is overfitting to the training prompts — you should reduce the number of steps  
D) reward_std near 0 means the reward function has a bug  

---

## Short Answer

**Q4.** Write the 5-part iteration log format (Hypothesis, Experiment, Metric, Result, Analysis) for the following scenario: "My v3 model generates correct SQL for simple queries but always generates `SELECT *` for queries involving aggregation, even though v2-dpo did not do this."

---

**Q5.** You want to test whether increasing K from 8 to 16 improves v3. Estimate the compute cost difference and the expected improvement in reward_std. When would you justify the 2× compute increase?

---

## Deep Scenario

**Q6.** You have run 3 experiments in Weeks 50–51:
- Exp 1: +250 complex prompts → complex query acc improved from 49% to 61%
- Exp 2: Reward fix (reduce level 0.2 to 0.05) → semantic acc improved from 55% to 59%
- Exp 3: Temperature increase (0.7 → 0.9) → reward_std recovered but mean_reward dropped from 0.44 to 0.38

You now need to pick ONE model checkpoint to submit for the Phase 5 Gate (Week 52). The gate requires: (1) v3 beats v1 on execution accuracy, (2) v3 has been produced by SFT + DPO + GRPO in sequence, (3) you can explain all three alignment methods mathematically.

Which checkpoint do you submit and why? How do you handle the tradeoff between exp 2 (better semantics) and exp 3 (degraded mean_reward)?
