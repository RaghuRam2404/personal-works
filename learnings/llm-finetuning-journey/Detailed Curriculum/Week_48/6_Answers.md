# Week 48 Quiz Answers

## Q1. Answer: B

**Answer:** B — Model collapsing — all completions scoring similarly, producing near-zero gradients.

**Why:** When reward_std ≈ 0.02, every completion in each group is receiving nearly the same reward. With GRPO's group normalization, all advantages become near zero: (r_i − mean) / 0.02 is numerically unstable and essentially zero for all i. The policy gradient update is effectively zero, meaning the model has stopped learning. The mean_reward increase from 0.18 to 0.24 suggests the model did improve early in training, but has now converged to a local behavior that consistently generates the same reward for all K completions (often happens when the model finds one reliable SQL pattern and repeats it).

**Fix:** Use harder prompts where the model succeeds 20–60% of the time (not 100% or 0%). Add more diversity to the prompt set and/or increase temperature from 0.7 to 0.9 to force more varied completions.

---

## Q2. Answer: B

**Answer:** B — Load the step-550 checkpoint, restore optimizer state, and continue from step 550.

**Why:** GRPO is an on-policy algorithm — it generates fresh rollouts at every step. A checkpoint at step 550 contains the model weights and optimizer state at that point. Resuming from step 550 is equivalent to having trained 550 steps and then continuing. The training distribution will be the same (same prompt dataset, same reward function). TRL's GRPOTrainer supports `resume_from_checkpoint=True` or a specific checkpoint path.

**Why others are wrong:**
- A: GRPO can resume mid-run — there is no technical reason to restart from 0.
- C: You cannot skip to step 600 without the checkpoint at step 600.
- D: The reference model is frozen and stateless — it does not accumulate state during training.

---

## Q3. Answer: C

**Answer:** C — The model developed chain-of-thought reasoning before SQL generation.

**Why:** This is the DeepSeek-R1 "aha moment" in your domain. GRPO with verifiable rewards (SQL execution) may spontaneously cause the model to generate reasoning steps before the final SQL, because completions with reasoning chains have a higher probability of being correct. The reward does not penalize length, so longer (reasoned) completions that score 1.0 get reinforced over short completions that score 0.2. The result is an 50–60 token increase in mean generation length.

This is actually a positive signal — if the generated reasoning is coherent (e.g., "-- I need to JOIN orders with customers to get total revenue..."), it indicates the model is reasoning about the SQL before generating it.

---

## Q4. KL at 8.5 Nats — Intervention Decision

KL of 8.5 nats is elevated but not catastrophic (catastrophic is typically > 20 nats). Given that v3 still outperforms v2, the model has not degenerated. However, the risks of leaving KL at 8.5 nats are:

1. **Continued drift:** If training continues, KL may keep growing past 10 nats, eventually causing reward hacking patterns to become entrenched.
2. **Reduced prompt diversity:** The model may have narrowed its output distribution significantly, which reduces the within-group variance (and reward_std may drop toward 0 later).
3. **Generalization risk:** The model is now far from the SFT reference — it may perform poorly on prompt types that were underrepresented in GRPO training.

**Recommended action:** Increase β from 0.05 to 0.1 for the remaining steps. This will slow down but not undo the drift. Do not restart — the model is performing well and a restart would lose 200 steps of training. At the end of training, evaluate whether the high-KL model generalizes appropriately to held-out test prompts.

---

## Q5. Fixing Complex Query Gap in Week 50

In Week 50 iteration:

**Step 1:** Rebuild the training prompt set with 40% complex queries (instead of 10%). Complex = 3+ table JOINs, CTEs, subqueries, window functions, TimescaleDB hyperfunctions.

**Step 2:** Run 500 additional GRPO steps starting from your v3 checkpoint, using only complex prompts. The simpler prompts can be reintroduced after 200 steps.

**Step 3:** The reward function does not need to change — execution correctness on complex queries is the right signal.

**Step 4:** Re-evaluate on the complex query tier specifically. Target: v3(improved) > v2 by ≥ 10pp on complex queries.

---

## Q6. Deep Scenario — v3 Lower Semantic Accuracy than v2

**Diagnosis:** The disconnect between higher execution accuracy (+2pp) and lower semantic accuracy (−2pp) means v3 generates more queries that execute without error but return wrong data. This is the exact reward hacking pattern where the reward function rewarded execution success too heavily relative to semantic correctness.

Root cause: your reward function likely returned 0.2 for "executes and returns rows" for a large fraction of the training distribution (because you did not always have reference SQL for exact-match comparison). The model optimized for level-0.2 reward (executes and returns some rows) rather than for level-1.0 reward (returns the correct rows). In practice: the model learned that `SELECT * FROM main_table` or similar broad queries often execute and get 0.2, which is better than the 0.0 it got for syntax errors before.

**Week 50 iteration plan:**

Reward change: Make reference SQL available for more training prompts. Expand the training prompt set to include reference SQL for at least 80% of prompts (up from the current coverage). This pushes more completions toward the 0.5 and 1.0 reward levels, making level-0.2 (unverified execution) a minority signal rather than the dominant one. Also: demote level-0.2 to 0.05 to reduce its attractiveness as a "hack" target.

Training data change: Add 200 additional training prompts where the "correct" query requires specific WHERE clauses or specific column selections (so `SELECT * FROM table` will execute but get 0.0 row-count match with the reference). These prompts specifically punish the "broad SELECT" hack because the reference output will not match broad queries.

After iteration, the target is: v3(improved) > v2 on BOTH execution accuracy AND semantic accuracy.
