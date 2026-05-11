# Week 46 Quiz Answers

## Q1. Answer: B

**Answer:** B — A_i = (r_i − mean(r_1,...,r_K)) / std(r_1,...,r_K).

**Why:** GRPO replaces the learned critic with the empirical mean of the group's rewards. Subtracting the mean centers the advantages around zero (same as baseline subtraction in REINFORCE). Dividing by std normalizes scale. The result is that completions better than the group average get positive advantages and completions worse than average get negative advantages — without training a separate critic network.

**Why others are wrong:**
- A: This is the PPO advantage; it requires a critic V.
- C: The reference model's completion is not used as the baseline in GRPO; the within-group mean is.
- D: Z is the partition function from DPO theory; it is not used in GRPO.

---

## Q2. Answer: B

**Answer:** B — The gradient is zero.

**Why:** If all K rewards are equal (all 1 or all 0), the group mean equals every reward, so every advantage is (r_i − mean)/std = 0/std = 0. The gradient of the PPO clip objective is A_i × ∇ log π, and with A_i = 0, the gradient is zero. This is the correct behavior: if all completions are equally good (or bad), there is no evidence about which direction to update the policy. The model does not learn from these cases, which is statistically appropriate.

---

## Q3. Answer: B

**Answer:** B — DPO requires pre-labeled preference data; for math, GRPO uses the verifiable ground-truth answer.

**Why:** Math problems have deterministic answers. You can verify correctness by comparing the model's extracted answer to the expected answer — no human labeler needed, no reward model needed. DPO requires a static dataset of preference pairs, which would have to be collected upfront. GRPO samples fresh rollouts at training time and evaluates them against the verifier, allowing the model to explore new reasoning paths as it improves. This online exploration is essential for the reasoning chain emergence.

---

## Q4. Answer: B

**Answer:** B — The model discovers that reasoning chains improve final answer accuracy; correct answers get higher rewards.

**Why:** The GRPO reward only checks the final boxed answer against the expected answer. There is no reward for intermediate reasoning. Yet the model learns to generate extended reasoning chains because: completions with reasoning chains have a higher probability of arriving at the correct final answer than completions without reasoning. The reward selects for correct answers, and correct answers increasingly come from completions that reasoned carefully. No explicit supervision of reasoning is needed.

---

## Q5. Answer: B

**Answer:** B — The model drifts far from the reference distribution and generates incoherent text or exploits the reward.

**Why:** The KL penalty is the anchor that keeps the policy from diverging catastrophically. Without it, the PPO objective would push the policy to maximize within-group advantages without any constraint on the language distribution. For verifiable rewards like SQL execution, the model might discover that certain degenerate patterns (always generating `SELECT 1` if that is what your test DB always returns as success) achieve reward=1 without being useful SQL. The KL penalty makes these degenerate solutions expensive because they would require moving far from the reference distribution.

---

## Q6. GRPO and the Verifiable Reward Connection

PPO's critic serves one purpose: estimate V(s_t), the expected future return from the current state, so that we can compute A_t = G_t − V(s_t). The critic needs to be learned because V(s_t) is not observable — you only observe the final reward at the end of the episode.

For SQL with a verifiable reward, the final reward is deterministic given (prompt, completion): you run the SQL on the database and get 1 or 0. The uncertainty that the critic tries to model (what is the expected reward for this partial sequence?) is reducible: with enough rollouts (K completions), you can estimate the expected reward from the empirical within-group mean. As K increases, the mean becomes a better estimate of the true expected reward.

The key: for a stochastic human-preference reward, the mean of K completions is not stable (different human raters give different scores for the same completion). For a deterministic verifiable reward, the mean is stable — the same completion always gets the same score. GRPO exploits this determinism to replace the critic with a Monte Carlo estimate.

---

## Q7. Increasing Training Signal When All-Same Rewards Dominate

**Intervention 1: Use a shaped reward instead of binary.** Replace the binary {0, 1} reward with a shaped reward that has more gradations:
- +1.0: executes correctly AND returns the right rows
- +0.5: executes but returns wrong row count
- +0.2: SQL parses without syntax error but fails at execution
- 0.0: syntax error

With shaped rewards, it is much less likely that all K completions get exactly the same reward. Even if all K execute, they may return different row counts, giving different partial credit scores.

**Intervention 2: Increase K or use curriculum prompting.** If 70% of prompts are either trivially easy (all K solve) or trivially hard (all K fail), your prompt set is poorly calibrated. Either: (a) increase K so that harder prompts have at least some variety in outcomes (K=16 vs K=8), or (b) curate your prompt set to focus on "medium difficulty" prompts — those where your current policy succeeds 20–80% of the time. Prompts with 50% success rate provide maximum variance in the group rewards and therefore maximum gradient signal.

---

## Q8. K=16 vs K=2 Argument

**Theoretical advantage of K=16:** The group mean and std estimates used for normalization are sample statistics. The variance of the sample mean is σ²/K. With K=2, the mean is estimated from 2 samples — extremely noisy. With K=16, the estimate is 8× more precise. A noisy baseline (mean from K=2) can increase rather than decrease gradient variance, defeating the purpose of GRPO.

**Worst case for K=2:** If one completion gets reward=1 and the other gets reward=0, the advantages are perfectly ±1 regardless of how "hard" the prompt was. This means prompts where the model has 99% probability of success and prompts where it has 50% probability of success produce the same gradient magnitude, despite having very different information content. K=16 naturally scales gradient magnitude with difficulty (harder prompts produce larger std, more varied advantages).

**Compute cost ratio:** GRPO cost scales roughly linearly with K for the generation step (K forward passes through the model). K=16 is 8× more compute than K=2. For a 7B model on an A100, this means a training step that takes 2 minutes at K=2 takes 16 minutes at K=16. For your SQL domain, K=16 is worth it because: (1) SQL execution is fast (each DB query takes < 1 second, a small fraction of the generation cost), (2) the signal quality improvement from better advantage estimates directly translates to faster convergence, likely requiring fewer training steps overall. Recommendation: start with K=8 as a compromise, increase to K=16 only if 70%+ of steps are zero-gradient.
