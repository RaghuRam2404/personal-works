# Week 50 Quiz Answers

## Q1. Answer: B

**Answer:** B — You changed two variables and cannot attribute the improvement or KL increase to either specific change.

**Why:** Changing β and the dataset simultaneously makes it impossible to know if the complex query improvement came from the more targeted prompts or from the looser KL constraint (β=0.02 allows more deviation from π_ref, which may help or hurt). Similarly, the KL increase to 9 nats might have been caused by β reduction — but you cannot be sure. Controlled experiments change exactly one thing. If you must change two things, run them sequentially (Experiment 1: dataset change only; Experiment 2: β change from the result of Experiment 1).

---

## Q2. Answer: B

**Answer:** B — Add 100 time_bucket prompts and run 300 steps from v3.

**Why:** This is the minimal targeted experiment that directly tests the hypothesis. Starting from v3 preserves all improvements made so far. 300 steps is enough to verify whether time_bucket prompts help — if they do, you can continue for more steps. Restarting from v2 (option A) throws away the GRPO improvements from Weeks 47–48. Option C (re-doing DPO) is expensive and would also require repeating Week 45. Option D tests whether the problem exists but does not fix it.

---

## Q3. Answer: B

**Answer:** B — The improvement came mostly from the first 100 steps; the last 200 produced near-zero gradient.

**Why:** If reward_std ≈ 0 for the last 200 steps, the model converged to generating nearly identical completions for every prompt (mode collapse). During this period, the policy gradient is essentially zero, so no learning occurred. The 5pp improvement came from the first 100 steps when the reward distribution was still varied. The last 200 steps were wasted compute — the model may have slightly overfit to the single reliable pattern it found, possibly degrading generalization. In a future run, stop training when reward_std drops below 0.05 for more than 50 consecutive steps.

---

## Q4. Iteration Log for SELECT * Bug

```
Hypothesis: v3 always generates SELECT * for aggregation queries because the GRPO 
  training set lacked aggregation prompts where SELECT * would be wrong. The model 
  found that SELECT * (which executes) reliably gets reward 0.2, and aggregation 
  queries never got reward 0.5 or 1.0 because reference SQL was not available.

Experiment: 
  1. Add reference SQL to 100 aggregation prompts in the training set.
  2. Change reward_fn: for aggregation prompts (detected by expected row count = 1), 
     return 0.0 for any query that does NOT include GROUP BY or an aggregate function (COUNT, SUM, etc.).
  3. Run 200 GRPO steps from v3 with the modified reward function on aggregation-heavy prompts.

Metric: % of v3-iter1's aggregation query completions that include GROUP BY or COUNT/SUM/AVG.

Result: [to be filled after running the experiment]

Analysis: [to be filled after running the experiment]
```

---

## Q5. K=8 vs K=16 Cost-Benefit Analysis

**Compute cost difference:** GRPO generation time scales nearly linearly with K (K=16 requires 2× more inference than K=8 per step). On an A100 80GB, if K=8 takes 90 seconds per step, K=16 takes ~170 seconds. For 300 steps: K=8 takes 7.5 hours; K=16 takes 14 hours. Cost differential: roughly 2× RunPod cost (~$15 vs $30 for 300 steps on A100 80GB).

**Expected improvement in reward_std:** With K=16, the within-group mean and std are estimated from 16 samples instead of 8, giving a more reliable advantage estimate (variance of sample mean is σ²/K, so 2× better with K=16). In practice, if your current reward_std is already healthy (0.2–0.4), K=16 gives marginal improvement. If reward_std is near 0 (mode collapse), K=16 does not fix the root cause.

**When to justify K=16:** When the mean reward is below 0.3 and reward_std is volatile (high variance between steps). This suggests the advantage estimates from K=8 are noisy, and a better baseline (from K=16) would stabilize training. Also justified when training on very hard prompts where the model succeeds only 10–20% of the time — K=16 guarantees at least 1–3 successful completions per group even at 10% success rate, whereas K=8 may produce groups with all-zero rewards.

---

## Q6. Checkpoint Selection for Phase 5 Gate

**Recommended checkpoint:** The model after Experiment 2 (not Experiment 3).

**Reasoning:** The Gate criteria are: (1) v3 beats v1 on execution accuracy, (2) SFT→DPO→GRPO pipeline, (3) mathematical understanding of all three methods. The key gate metric is execution accuracy vs v1. The model after Experiment 2 has the highest semantic accuracy (59%) and good execution accuracy. Experiment 3 improved reward_std during training but decreased mean_reward (0.44 → 0.38), which likely means the higher temperature caused more diverse but less reliable completions — potentially hurting execution accuracy.

**The tradeoff:** Experiment 2 vs Experiment 3 is a tradeoff between reliability (exp 2: consistent 0.5-reward completions) and diversity (exp 3: more varied completions but lower average quality). For the Gate, reliability matters more — you need to demonstrate the model actually works. The training metric (mean_reward) is a proxy for quality; if exp 3 dropped mean_reward, it likely dropped real-world execution accuracy.

**Final answer:** Submit the Experiment 2 checkpoint. In the Gate submission, document all 3 experiments honestly — showing that you ran Experiment 3 and it hurt is just as valuable a finding as showing Experiment 2 helped. The Gate is not just about the best model; it is about demonstrating the ability to iterate intelligently.
