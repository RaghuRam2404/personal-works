# Week 52 Gate — Quiz Answers

## Q1. Answer: B

**Answer:** B — Z(x) is the same constant for both y_w and y_l, so it cancels in the difference.

**Why:** Z(x) = Σ_y π_ref(y|x)·exp(r(x,y)/β) is a sum over all possible completions y for prompt x. It does not depend on any specific completion. When you compute r(x, y_w) − r(x, y_l), the β·log Z(x) term appears in both and subtracts out: (β·log(π*(y_w|x)/π_ref(y_w|x)) + β·log Z(x)) − (β·log(π*(y_l|x)/π_ref(y_l|x)) + β·log Z(x)) = β·log(π*(y_w|x)/π_ref(y_w|x)) − β·log(π*(y_l|x)/π_ref(y_l|x)). This exact cancellation is what makes DPO tractable.

---

## Q2. Answer: B

**Answer:** B — Rejected examples contain refusal patterns; DPO reduced their probability alongside valid rejected responses.

**Why:** DPO decreases the log-probability of all tokens in rejected completions. If your Week 44 preference dataset included any rejected examples that began with "I cannot" (e.g., a model that failed to generate SQL and instead generated a refusal was labeled as "rejected"), DPO trained the model to suppress the probability of starting a response with "I cannot." This inadvertently reduced the probability of generating legitimate refusals for ambiguous prompts. Fix: audit the rejected examples for non-SQL text and remove those pairs from the dataset.

---

## Q3. Answer: C

**Answer:** C — GRPO is optimal for this setting.

**Why:** The setting has all three conditions that make GRPO the clear winner: (1) verifiable reward (SQL execution) means no human labeler or reward model is needed, (2) online generation in GRPO provides fresh signals at every step, and (3) GRPO's no-critic design means 2× memory instead of PPO's 4× — fitting within the 32GB budget. PPO (A) would require a critic and reward model, using all 32GB. DPO (B) is offline — it cannot use fresh execution feedback at training time. KTO (D) requires a pre-built labeled dataset of good/bad SQL pairs, not online execution.

---

## Q4. Answer: B

**Answer:** B — Zero. All advantages are 0 when all rewards are equal.

**Why:** A_i = (1.0 − mean([1,1,1,1,1,1,1,1])) / std([1,1,1,1,1,1,1,1]) = (1.0 − 1.0) / 0 → numerically handled as 0/(std + 1e-8) ≈ 0. The policy gradient is A_i · ∇ log π, and with A_i = 0, the gradient is zero. This is statistically correct — if all K completions succeed, there is no evidence distinguishing which SQL structure is better. No update is needed, and the model should not change.

---

## Q5. Answer: C

**Answer:** C — Increase generation temperature from 0.7 to 0.9.

**Why:** reward_std → 0 means the model is generating nearly identical completions for each prompt (mode collapse). The most direct fix is to increase the temperature during the GRPO rollout generation step. Higher temperature forces more diverse completions, increasing the within-group variance of rewards and restoring a non-zero reward_std. Increasing β (A) would not fix mode collapse — it addresses KL drift, not diversity. Switching to DPO (B) abandons the GRPO progress. Reducing K (D) worsens the problem — fewer completions means even more likely that all K are identical.

---

## Q6. GRPO Advantages Calculation

Rewards = [1, 0, 1, 0]:

```
mean = (1 + 0 + 1 + 0) / 4 = 0.5

sample variance (ddof=1):
  Σ(r_i - mean)² = (0.5² + 0.5² + 0.5² + 0.5²) = 4 × 0.25 = 1.0
  var = 1.0 / (4-1) = 1/3
  std = sqrt(1/3) ≈ 0.5774

advantages:
  A_1 = (1 - 0.5) / 0.5774 = 0.5 / 0.5774 ≈ 0.866
  A_2 = (0 - 0.5) / 0.5774 = -0.5 / 0.5774 ≈ -0.866
  A_3 = (1 - 0.5) / 0.5774 ≈ 0.866
  A_4 = (0 - 0.5) / 0.5774 ≈ -0.866
```

Result: [0.866, −0.866, 0.866, −0.866] ✓

---

## Q7. GRPO vs PPO — Why No Critic

(1) **What the critic does in PPO:** The critic (value network V(s_t)) estimates the expected future return from state s_t — the token prefix generated so far. This estimate is needed to compute the advantage A_t = G_t − V(s_t), which reduces gradient variance compared to using the raw return.

(2) **What replaces it in GRPO:** The within-group empirical mean of rewards: mean(r_1, ..., r_K). This serves as the baseline exactly as V(s_t) does in REINFORCE baseline subtraction — it reduces gradient variance without introducing bias.

(3) **Why the replacement is valid for SQL specifically:** SQL execution is deterministic — the same SQL query on the same database always returns the same result (reward). Therefore, the within-group mean is a reliable, unbiased estimate of the true expected reward E[r|prompt, model]. With K=8 deterministic samples, the estimate is stable. The critic in PPO would add no information that the deterministic execution result already provides.

(4) **When the replacement would NOT be valid:** When rewards are stochastic — e.g., different human raters giving different scores for the same completion. In this case, the within-group mean from K=8 samples is a noisy estimate of E[r|prompt], with variance σ²/K that may be too high for stable training. The learned critic in PPO can reduce this variance more effectively because it is trained on many (prompt, reward) pairs across all batches, not just the 8 samples in one group.

---

## Q8. SQL Reward Function Design

```
Level 0.0: 
  - SQL extraction fails (model generated no SQL)
  - SQL contains information_schema, pg_catalog, or pg_stat (anti-hack guard)
  - SQL does not start with SELECT or WITH
  - SQL has a syntax error

Level 0.1:
  - SQL parses correctly but fails at execution (runtime error: wrong table/column name)
  - SQL executes but returns 0 rows (likely wrong WHERE clause)

Level 0.2:
  - SQL executes without error AND returns at least 1 row
  - But no reference output available OR row count does not match expected
  - Anti-hack guard: reject if row_count > max(5 × expected_count, 10)

Level 0.5:
  - SQL executes AND row count exactly equals expected_count (± 1)
  - Values may differ (structural correctness)

Level 1.0:
  - SQL executes AND sorted(rows_actual) == sorted(rows_expected) (exact match)

Reasoning bonus (+0.05, maximum 5% of base reward 1.0):
  - Applied only when base reward >= 0.5 (not for failing queries)
  - Condition: len(text_before_sql) > 80 characters (proxy for reasoning chain)
  - Cap: bonus is never larger than 0.05

Timeout: SET statement_timeout = 500ms (500ms per query, enforced at DB level)
```

---

## Q9. Skip DPO in Phase 6?

**The colleague's argument has merit when:** Phase 5's DPO produced no improvement on complex queries, which is true. If complex query performance is the primary goal of Phase 6, and you now have reference SQL for semantic verification in the GRPO reward function, GRPO alone might be sufficient to get both easy query syntax-error reduction AND complex query improvement.

**The colleague is wrong when:** DPO provides benefits GRPO does not — specifically, reducing syntax error rate from ~22% to ~7%. GRPO's sparse reward at the end of generation means the model gets no gradient for "you had the right JOIN but a syntax error in the WHERE clause." DPO's token-level log-probability optimization directly reduces the probability of known error patterns (syntax errors, wrong column names seen in rejected examples). Removing DPO risks bringing the syntax error rate back up, which GRPO may not fully correct because GRPO's reward is binary (executes/doesn't) and syntax errors get reward=0 — same as a reasonable SQL that happens to fail on your test DB.

**What to measure to decide:** Run one Phase 6 training branch with SFT → GRPO (skipping DPO) and compare to SFT → DPO → GRPO at the same number of total training steps. Specifically measure: (1) syntax error rate at 200 GRPO steps (can GRPO alone reduce syntax errors as fast as DPO?), (2) complex query accuracy at 500 GRPO steps for both paths. If SFT → GRPO achieves the same syntax error rate as SFT → DPO → GRPO and similar complex query performance, the colleague is right and DPO can be dropped from Phase 6.
