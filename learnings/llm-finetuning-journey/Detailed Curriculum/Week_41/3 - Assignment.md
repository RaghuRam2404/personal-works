# Week 41 Assignment — Implement REINFORCE and Solve CartPole

## Setup Checklist

- [ ] Google Colab (Free tier is sufficient; CartPole is CPU-only)
- [ ] Python packages: `torch`, `gymnasium` (the maintained fork of `gym`)
- [ ] No GPU needed — CartPole environment runs on CPU in seconds

```bash
pip install torch gymnasium
```

---

## Task 1 — Implement REINFORCE from Scratch

**Goal:** Write a working policy gradient agent that solves CartPole-v1 using only PyTorch. No RL libraries (no Stable Baselines, no RLlib, no TorchRL).

**Requirements:**
- Define a simple MLP policy network: input = 4 (CartPole state dim), hidden = 128, output = 2 (left/right)
- Policy outputs logits; sample actions using `torch.distributions.Categorical`
- Implement the episode rollout loop: collect (state, action, reward) tuples for one complete episode
- Compute discounted returns G_t = Σ_{k≥t} γ^k r_k with γ = 0.99
- Normalize returns: subtract mean, divide by std (this reduces variance significantly)
- Compute the REINFORCE loss: `-log_prob * G_t` summed over all steps (remember: PyTorch minimizes, so negate)
- Train for at least 800 episodes
- Log episode reward every 50 episodes
- Solve criterion: achieve average reward ≥ 195 over 100 consecutive episodes

**Deliverable:** `week-41-reinforce/cartpole_reinforce.py`. GitHub commit message: `week-41-reinforce-cartpole`.

**Hints:**
- `torch.distributions.Categorical(logits=...)` gives you `.log_prob(action)` and `.sample()` in one object
- Store `log_prob` during rollout before zeroing gradients — do not call `optimizer.zero_grad()` inside the rollout loop
- The return normalization (`returns = (returns - returns.mean()) / (returns.std() + 1e-8)`) is not optional — without it, the agent often fails to converge on CartPole

---

## Task 2 — Map the MDP to a Language Model

**Goal:** Write a 400–600 word Markdown document (`week-41-reinforce/llm_rl_mapping.md`) that maps each component of the CartPole MDP to the equivalent in LLM RLHF.

**Requirements:**
- Cover: state space, action space, policy, reward, episode, terminal condition, discount factor
- For each component: 2–3 sentences explaining what it corresponds to in the LLM setting AND why
- Include one concrete example: "For the prompt 'SELECT all users', the state at step 3 might be [SELECT, all, users, FROM] — a partial SQL generation."
- Discuss the credit assignment problem: why is it harder in the LLM setting than CartPole?

**Deliverable:** `week-41-reinforce/llm_rl_mapping.md`

---

## Task 3 — Derive the Policy Gradient on Paper

**Goal:** Write out the full derivation of the policy gradient theorem by hand.

**Requirements:**
- Start from J(θ) = E_{τ∼π_θ}[R(τ)]
- Show how the log-derivative trick converts ∇_θ p(τ|θ) into p(τ|θ) · ∇_θ log p(τ|θ)
- Arrive at the REINFORCE gradient estimator
- Take a photo of your handwritten derivation and include it as `week-41-reinforce/pg_derivation.jpg` (or write it as LaTeX/Markdown math)

**Deliverable:** `week-41-reinforce/pg_derivation.jpg` or `pg_derivation.md`

---

## Stretch Goals

- Add a learned baseline (value network) to reduce variance. The value network takes the state and outputs a scalar V(s). Subtract V(s_t) from G_t before computing the loss. Does it converge faster?
- Plot the learning curve: episode reward vs. episode number. Observe the classic CartPole "hockey stick" curve — random wandering for ~200 episodes, then rapid improvement.
- Try γ = 0.9 vs γ = 0.99 vs γ = 1.0. What happens?
