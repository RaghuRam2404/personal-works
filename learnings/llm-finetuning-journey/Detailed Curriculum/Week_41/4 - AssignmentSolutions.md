# Week 41 Assignment Solutions

## Task 1 — REINFORCE Key Snippets

The trickiest part is the rollout loop and the return computation. Here is the core:

```python
import torch
import torch.nn as nn
import gymnasium as gym
from torch.distributions import Categorical

class PolicyNet(nn.Module):
    def __init__(self, obs_dim=4, hidden=128, n_actions=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )
    def forward(self, x):
        return self.net(x)

def compute_returns(rewards, gamma=0.99):
    G, returns = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    # Normalize
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns

# Training loop skeleton
env = gym.make("CartPole-v1")
policy = PolicyNet()
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)

for episode in range(1000):
    obs, _ = env.reset()
    log_probs, rewards = [], []
    done = False
    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32)
        logits = policy(obs_t)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_probs.append(dist.log_prob(action))
        obs, reward, terminated, truncated, _ = env.step(action.item())
        rewards.append(reward)
        done = terminated or truncated

    returns = compute_returns(rewards)
    loss = -torch.stack(log_probs).dot(returns)  # scalar
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**Expected output:** Episode reward should hover around 10–30 for the first 200 episodes, then begin climbing toward 200+. Typical solve (mean ≥ 195 over 100 episodes) happens between episodes 400 and 700.

**Common gotchas:**
- Forgetting to detach `log_probs` from the computation graph before stacking — do not call `.detach()`, but do not add rewards to the graph either
- Using `gym` instead of `gymnasium` — the API changed: `env.reset()` now returns `(obs, info)` not just `obs`
- Not normalizing returns: the agent often stalls at reward ~10 permanently
- Using `optimizer.zero_grad()` inside the rollout loop — this kills stored gradients
- `dist.log_prob()` returns a tensor with grad_fn; make sure you do not break the graph by converting to numpy mid-loop

---

## Task 2 — LLM RL Mapping (Model Answer Summary)

Key points your document should cover:

- State = prefix (prompt + partial generation). In CartPole it is 4 floats; in LLM it is tens of thousands of floats (the KV cache or the full embedding).
- Action = next token. CartPole has 2 actions; a 7B LLM has ~32,000 actions (vocabulary size). This makes the credit assignment and variance problems much worse.
- Episode = one complete generation from start to EOS token.
- Reward is sparse: reward model returns a scalar only at EOS. In CartPole, every step gives +1.
- Discount γ: often set to 1.0 in LLM RLHF (or the KL penalty takes its place as a regularizer).

---

## Task 3 — Policy Gradient Derivation (Key Steps)

Starting from J(θ) = Σ_τ p(τ|θ) R(τ):

```
∇_θ J(θ) = Σ_τ ∇_θ p(τ|θ) R(τ)
           = Σ_τ p(τ|θ) [∇_θ log p(τ|θ)] R(τ)   # log-derivative trick
           = E_{τ∼π_θ} [∇_θ log p(τ|θ) · R(τ)]
```

Since log p(τ|θ) = Σ_t log π_θ(a_t|s_t) + const (transition probs cancel in gradient):

```
∇_θ J(θ) = E_{τ∼π_θ} [ Σ_t ∇_θ log π_θ(a_t|s_t) · G_t ]
```

---

## How to Verify You Did It Right

1. Run 5 seeds. At least 3 should solve CartPole (mean reward ≥ 195) by episode 800.
2. Print the return normalization sanity check: `print(returns.mean().item(), returns.std().item())` — should be ~0 and ~1 immediately after normalization.
3. Check loss is decreasing on average over 100-episode windows (not monotonically, but trending down).
4. Can you state, without notes: "The policy gradient theorem says the gradient of expected return equals the expectation of the product of ∇ log π and the return"?
