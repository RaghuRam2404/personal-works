# Week 41 Quiz Answers

## Q1. Answer: B

**Answer:** B — A single token selected from the vocabulary.

**Why:** Language generation is a sequential decision process. At each decoding step, the model selects one token from the vocabulary (the action). The episode is the complete sequence from the first generated token to EOS.

**Why others are wrong:**
- A: The entire completion is the trajectory, not a single action.
- C: The reward model score is the reward signal, not an action.
- D: Gradient updates happen outside the MDP; they are the learning mechanism, not part of the environment.

---

## Q2. Answer: B

**Answer:** B — p(τ|θ) · ∇_θ log p(τ|θ).

**Why:** The log-derivative trick uses the identity ∇_θ p = p · ∇_θ log p. This is derived by noting that ∇_θ log p = (1/p) · ∇_θ p, so ∇_θ p = p · ∇_θ log p. The trick converts a gradient of a probability into an expectation under that probability, which can be estimated by sampling.

**Why others are wrong:**
- A: Missing the gradient; this is just a weighted probability.
- C: R(τ) is not differentiated — the reward function is not a function of θ.
- D: Algebraically equivalent to B but incorrectly written as a ratio rather than a product.

---

## Q3. Answer: B

**Answer:** B — The gradient variance is extremely high because all 200 steps get the same return G_0.

**Why:** With sparse reward and no discounting (γ≈1), every action in the episode receives the return for the entire episode. The gradient signal tells all 200 token choices "you were responsible for this outcome" equally, which is wrong and introduces enormous variance. The agent cannot determine which early tokens were responsible for success or failure.

**Why others are wrong:**
- A: REINFORCE can learn from sparse reward — it is just inefficient and high-variance.
- C: γ = 0 would zero out all rewards; this would prevent learning entirely.
- D: The Markov property is about transition dynamics, not reward density.

---

## Q4. Answer: C

**Answer:** C — On-policy methods update using data sampled from the current policy; off-policy methods can use data from older policies.

**Why:** This is the defining distinction. REINFORCE and PPO are on-policy — you must generate fresh episodes from π_θ before each update. DQN is off-policy — it stores experience in a replay buffer and samples from it even after the policy has changed.

**Why others are wrong:**
- A: This has it backwards — off-policy methods use replay buffers.
- B: Off-policy methods (e.g., SAC) handle continuous action spaces.
- D: On-policy methods can use a critic (PPO does) but it is not the defining characteristic.

---

## Q5. Answer: C

**Answer:** C — Reducing gradient variance without introducing bias.

**Why:** Subtracting any baseline b(s_t) from the return gives an unbiased estimator as long as b does not depend on the action. The mean of returns is a state-independent baseline, so it is valid. The practical effect is that returns which are "above average" get a positive weight and "below average" get a negative weight, which produces a much cleaner learning signal.

**Why others are wrong:**
- A: Return normalization has nothing to do with the discount factor.
- B: This is not mathematically the advantage unless the baseline is V(s_t) — the mean of G_t is not the value function.
- D: Convergence speed varies; there is no guarantee of solving in 500 episodes.

---

## Q6. REINFORCE Loss — Model Answer

```python
log_probs, rewards = [], []
obs, _ = env.reset()
done = False
while not done:
    logits = policy(torch.tensor(obs, dtype=torch.float32))
    dist = Categorical(logits=logits)
    action = dist.sample()
    log_probs.append(dist.log_prob(action))  # collect during rollout
    obs, r, term, trunc, _ = env.step(action.item())
    rewards.append(r)
    done = term or trunc

# Compute discounted returns
G, returns = 0.0, []
for r in reversed(rewards):
    G = r + 0.99 * G
    returns.insert(0, G)
returns = torch.tensor(returns)
returns = (returns - returns.mean()) / (returns.std() + 1e-8)

# REINFORCE loss (negative because we maximize J, but PyTorch minimizes)
loss = -(torch.stack(log_probs) * returns).sum()
optimizer.zero_grad(); loss.backward(); optimizer.step()
```

---

## Q7. Advantage Function — Model Answer

In REINFORCE without a baseline, G_t acts as the return. With baseline subtraction using V(s_t), the quantity (G_t − V(s_t)) is called the advantage. Practically, this centers the learning signal around zero: actions better than average get a positive update, actions worse than average get a negative update. Without a baseline, all actions in a good episode get positive updates regardless of whether individual actions were good, which introduces high variance. The advantage makes credit assignment more precise.

---

## Q8. DPO and RL Theory — Model Answer

The statement is partially true but misleading. DPO does not run a RL training loop — there is no rollout, no reward model scoring completions at training time, and no PPO update. However, DPO is derived by solving the same KL-constrained RL objective that RLHF/PPO optimizes:

max_π E[r(x,y)] − β · KL[π || π_ref]

DPO derives a closed-form optimal policy for this objective and then re-parameterizes the reward in terms of policy log-ratios. The resulting loss looks like a classification loss on preference pairs, but its correctness depends on the RL derivation in the appendix. Without understanding the KL-constrained RL problem, you cannot understand why the DPO loss converges to the same solution PPO would — or why it can fail when its assumptions are violated.

---

## Q9. Scenario — REINFORCE Stuck at 8% SQL Execution Rate

**Intervention 1 (highest impact): Switch to a dense/shaped reward.**
Giving +1 only for correct execution and 0 otherwise is extremely sparse. 92% of episodes get no gradient signal whatsoever. Add partial rewards: +0.3 for valid SQL syntax, +0.1 for correct table names, +0.5 for correct columns before execution. This gives the model gradient signal even for failed queries, drastically reducing effective sparsity.

**Intervention 2: Add a warm start with your SFT model.**
REINFORCE starting from a base model has to discover the entire SQL syntax space via random exploration. If you initialize from your SFT model (which already generates ~30–40% valid SQL), the starting policy is much stronger. The RL training then improves an already-functional policy rather than searching from scratch. This is exactly what the InstructGPT pipeline does: SFT first, then RL.

**Intervention 3: Increase batch size / use multiple rollouts per update.**
REINFORCE with a single episode per update has catastrophically high gradient variance. Collect 16 or 32 episodes per parameter update and average the gradient. With 8% success rate, you need many episodes just to see a few successes. This is also why GRPO (Week 46) uses group sizes of 8–16 completions per prompt — it directly addresses this variance problem.
