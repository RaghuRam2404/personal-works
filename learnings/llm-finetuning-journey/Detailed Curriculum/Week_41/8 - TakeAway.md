# Week 41 TakeAway — RL Primer

**One-liner:** The policy gradient theorem lets you optimize non-differentiable rewards via log-probability weighting.

---

## Key Formulas

```
# Policy gradient theorem
∇_θ J(θ) = E_{τ~π_θ} [ Σ_t ∇_θ log π_θ(a_t|s_t) · G_t ]

# Discounted return
G_t = Σ_{k≥t} γ^(k-t) · r_k

# Advantage (baseline subtraction)
A_t = G_t - V(s_t)

# REINFORCE loss (PyTorch convention: minimize negative)
loss = -(log_probs * returns).sum()
```

---

## Key Code Pattern

```python
# Full REINFORCE episode
log_probs, rewards = [], []
obs, _ = env.reset()
while not done:
    dist = Categorical(logits=policy(obs_tensor))
    action = dist.sample()
    log_probs.append(dist.log_prob(action))
    obs, r, term, trunc, _ = env.step(action.item())
    rewards.append(r)
    done = term or trunc

# Compute + normalize returns
G, returns = 0, []
for r in reversed(rewards): G = r + 0.99*G; returns.insert(0, G)
returns = torch.tensor(returns)
returns = (returns - returns.mean()) / (returns.std() + 1e-8)

loss = -(torch.stack(log_probs) * returns).sum()
optimizer.zero_grad(); loss.backward(); optimizer.step()
```

---

## LLM Mapping (memorize this table)

| RL | LLM |
|---|---|
| State s_t | prompt + tokens so far |
| Action a_t | next token (vocab ~32K) |
| Policy π_θ | LLM softmax |
| Episode | full generation to EOS |
| Reward | reward model score at EOS |
| Sparse reward | reward only at final token |

---

## Decision Rules

- If reward is sparse → add intermediate shaped rewards or use a strong SFT initialization before RL
- If gradient variance is high → add baseline subtraction (value function) or increase batch size
- If you need non-differentiable rewards → policy gradient works; you do not need to differentiate through the reward
- On-policy (REINFORCE/PPO) → fresh rollouts every update; Off-policy (DQN) → replay buffer

---

## Numbers to Remember

- CartPole solve = mean reward ≥ 195 over 100 episodes
- Typical γ = 0.99 for episodic tasks; LLM RLHF often uses γ = 1.0 + KL penalty
- Return normalization: subtract mean, divide by std + 1e-8 (epsilon prevents division by zero)
- REINFORCE needs 400–700 episodes to solve CartPole with return normalization

---

## Red Flags

- Loss not decreasing after 200 episodes: check return normalization is applied
- All rewards are 0 in 90%+ of episodes: reward is too sparse; add shaping
- Loss is NaN: log_prob of zero-probability action; add entropy regularization
- Reward plateaus and never improves: learning rate too high or too low; try 1e-3, 1e-2, 3e-2
