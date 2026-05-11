# Week 42 Quiz Answers

## Q1. Answer: B

**Answer:** B — It prevents the new policy from assigning much higher probability to the action than the old policy.

**Why:** When A_t > 0, we want to increase the probability of action a_t. The unclipped objective would do this without limit. The clip at 1+ε caps the ratio, preventing a very large probability increase in one update. The min() then selects the lower of the clipped and unclipped values, making the objective conservative.

**Why others are wrong:**
- A: When A_t > 0, we want to increase probability, not prevent decreases. Preventing decreases matters when A_t < 0.
- C: The clip acts on the policy ratio, not the reward or KL budget.
- D: Clipping applies in both advantage signs, just for different reasons.

---

## Q2. Answer: A

**Answer:** A — The model forgets how to generate coherent text because there is no regularization.

**Why:** The KL penalty keeps the policy anchored to the SFT model. Without it, PPO optimizes the reward model score without any constraint on language quality. The policy may quickly find that certain degenerate patterns (repetition, gibberish, or specific exploits) score highly on the reward model. This is reward hacking, and it degrades generation quality rapidly.

**Why others are wrong:**
- B: Speed does increase temporarily, but quality collapses; convergence to a useful policy fails.
- C: The value function is defined regardless of β.
- D: Collapse to a single token is not the typical failure mode; reward hacking patterns are usually more structured.

---

## Q3. Answer: B

**Answer:** B — 1-step TD residual (low variance, high bias).

**Why:** GAE with λ=0 gives A_t = δ_t = r_t + γ·V(s_{t+1}) − V(s_t), the single-step TD error. This has low variance (only one step of randomness) but high bias if V is inaccurate. At λ=1, GAE gives the full Monte Carlo return minus the baseline, which has zero bias but high variance.

---

## Q4. Answer: B

**Answer:** B — A frozen copy of the SFT model used only to compute KL divergence.

**Why:** The reference model is frozen throughout training. Its role is to define the "starting point" policy — the trained model is penalized for diverging too far from it. If it were updated, the KL would be computed against a moving target and would lose its regularizing effect.

**Why others are wrong:**
- A: Slow-moving average is used in SAC/DDPG target networks, not in RLHF reference models.
- C: The reward model is separate from the reference model.
- D: The critic estimates V(s); the reference model computes π_ref(a|s). These are different objects.

---

## Q5. PPO Reward Formulation

The total reward at each position t in a generation of length T:

```
r_t = -β · KL_t(π_θ || π_ref)              for t < T (intermediate tokens)
r_T = r_RM(x, y) - β · KL_T(π_θ || π_ref)  for t = T (final token)
```

The reward model score r_RM is applied only at the final token (EOS position) because the reward model scores the complete generation, not partial generations. The KL penalty is applied at every token position because the policy makes a probability decision at every step, and we want to regularize every step, not just the final one. This means longer generations pay a larger KL penalty in aggregate, which implicitly encourages conciseness.

---

## Q6. Four Forward Passes

1. **Actor (training model):** Generates the completion and computes log π_θ(a_t|s_t) for the PPO loss. Must be differentiable (gradients flow through it).
2. **Critic (value model):** Estimates V(s_t) at each position, used for GAE advantage computation. Also must be differentiable; value loss updates it.
3. **Reward model:** Scores the completed generation. Frozen; provides the terminal scalar reward. Cannot be eliminated unless you replace RM with a verifiable reward (GRPO's key insight).
4. **Reference model:** Computes π_ref(a_t|s_t) for KL penalty computation. Frozen. Cannot be eliminated without removing the anchor that prevents reward hacking.

The actor and critic are often the same model with two output heads to save memory, but they conceptually serve different functions.

---

## Q7. Reward Hacking in SQL

The phenomenon is **reward hacking** (also called "Goodhart's Law in RL"). The reward model has been exploited — the policy has found patterns that score highly on the RM but do not correspond to actual quality.

**Intervention 1:** Increase the KL penalty coefficient β. Currently the policy is drifting too far from the SFT model (which generated valid SQL). A larger β will anchor the policy closer to the SFT baseline, which already has decent SQL quality.

**Intervention 2:** Augment the reward with an execution-based signal. If you add a binary +1/0 reward for whether the SQL actually executes on a real Postgres database, reward hacking becomes much harder — no syntactically broken or hallucinated SQL can exploit this. This is exactly what Week 44 builds toward.

---

## Q8. Deep Scenario — PPO Failure Diagnosis

**KL at 45 nats:** The policy has drifted catastrophically from the reference model. Normal RLHF runs stay below 5–10 nats. This means β is too small or was not set. Immediate fix: increase β from current value (likely 0.01–0.05) to 0.2–0.5 and restart from the last checkpoint before KL exploded.

**RM score at 8.2/10 with degenerate outputs:** Classic reward hacking. The reward model has been exploited. The `WHERE 1=1` pattern likely scores high because the reward model was trained on data where such patterns appeared in valid SQL. Fix: add an execution-based reward component (SQL must execute on real DB) as a hard filter or additional term. Also reduce the number of PPO epochs per batch (from 4 to 1 or 2) to slow down the exploitation.

**Value loss oscillating wildly:** The value function has diverged from the true value because the policy distribution has changed so rapidly. Fix: (1) reduce PPO learning rate by 10x, (2) clip gradient norm to 0.5, (3) consider reinitializing the value head from the current frozen actor weights. Also, check that the value function is being trained with a separate, lower learning rate than the actor.

**Concrete hyperparameter adjustments:**
- β (KL coefficient): 0.01 → 0.3
- PPO epochs per batch: 4 → 1  
- Learning rate: 1e-5 → 1e-6
- Gradient clipping: add `max_grad_norm=0.5`
- Add execution reward: `r_total = r_RM + r_exec` where r_exec ∈ {0, 1}
