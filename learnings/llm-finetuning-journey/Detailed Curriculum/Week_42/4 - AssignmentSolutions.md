# Week 42 Assignment Solutions

## Task 1 — PPOTrainer Annotation Key Points

The six critical locations to find and annotate in TRL's PPOTrainer:

**1. Reference model KL computation**
Look for something like:
```python
# In compute_rewards() or similar method
ref_logprobs = self.model.pretrained_model(...)  # frozen reference model
kl = (logprobs - ref_logprobs)                  # per-token KL approx
# NOTE: This is the per-token KL approximation: E[log p - log q] ≈ KL(p||q)
# It is an approximation because exact KL requires summing over the full vocabulary
```

**2. Reward + KL combination**
```python
# rewards[i] = rm_score - kl_penalty
# The reward model score comes in as a scalar at the final token;
# the KL penalty is subtracted at every token position.
# This per-token KL is why long generations are penalized more than short ones.
non_score_reward = -self.kl_ctl.value * kl
rewards = non_score_reward + score  # score = RM score at EOS only
```

**3. GAE computation**
```python
# Backward pass over timesteps — look for reversed() or decrementing index
lastgaelam = 0
for t in reversed(range(gen_len)):
    delta = rewards[t] + gamma * values[t+1] - values[t]
    lastgaelam = delta + gamma * lam * lastgaelam
    advantages[t] = lastgaelam
returns = advantages + values  # returns = advantages + value baseline
```

**4. PPO clipping loss**
```python
ratio = torch.exp(logprobs - old_logprobs)  # π_new / π_old
pg_losses = -advantages * ratio
pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
# Take the pessimistic (max) of the two — this is the PPO clip
pg_loss = torch.mean(torch.max(pg_losses, pg_losses2))
```

**Expected output:** After annotating, you should be able to answer: "Where does the KL penalty go?" Answer: it is subtracted from the reward at each token position, so the total reward is `RM_score_at_EOS - β * Σ_t KL_t`.

**Common gotchas:**
- TRL versions change frequently; method names shift. Use `git log` to check when the file was last substantially modified.
- The KL approximation used (`logp − logp_ref`) is the first-order approximation of KL, not exact KL. It can be negative.
- Values (critic outputs) and logprobs (actor outputs) come from potentially different model heads — do not confuse them.
- The `old_logprobs` are computed before the PPO update epoch begins and held fixed; this is the "proximal" in PPO.
- TRL may normalize advantages before the clipping loss — check if `whitening(advantages)` is called.

---

## Task 3 — GAE Implementation

```python
import torch

def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """
    rewards: list of floats, length T
    values:  tensor of shape (T+1,) — includes bootstrap value at T
    Returns: advantages (T,), returns (T,)
    """
    T = len(rewards)
    advantages = torch.zeros(T)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t+1].item() - values[t].item()
        lastgaelam = delta + gamma * lam * lastgaelam
        advantages[t] = lastgaelam
    returns = advantages + values[:T]
    return advantages, returns

# Unit test
rewards = [1.0]
values = torch.tensor([0.5, 0.0])  # V(s0)=0.5, bootstrap V(s1)=0 (terminal)
adv, ret = compute_gae(rewards, values, gamma=1.0, lam=1.0)
assert abs(adv[0].item() - 0.5) < 1e-5, f"Expected 0.5, got {adv[0].item()}"
print("GAE unit test passed.")
```

---

## How to Verify You Did It Right

1. Your annotated file should have at least 20 lines of comments. Less means you skimmed.
2. For the GAE unit test: single step, r=1.0, V(s)=0.5, γ=λ=1.0 → δ = 1.0 + 0 − 0.5 = 0.5, advantage = 0.5. Passes? Good.
3. For Task 2: can you explain in one sentence per stage what the SQL equivalent is? If not, re-read the InstructGPT paper sections 3.1–3.3.
4. Can you state: "The PPO clip prevents the policy update from changing the probability ratio beyond 1±ε. The min() ensures we take the more pessimistic bound in both the positive and negative advantage cases."?
