# Week 46 TakeAway — GRPO and RLVR

**One-liner:** GRPO replaces the PPO critic with within-group reward normalization; verifiable rewards make the group mean an exact baseline.

---

## GRPO Algorithm (6 steps)

```
For each prompt x:
1. Generate K completions: {y_1,...,y_K} ~ π_θ
2. Compute rewards: {r_1,...,r_K} via verifier (SQL executor, test runner, etc.)
3. Normalize advantages:
   A_i = (r_i - mean(r)) / (std(r) + 1e-8)
4. Compute PPO clip loss with advantages A_i
5. Add KL penalty: -β · KL(π_θ || π_ref)  (prevents reward hacking)
6. Gradient step
```

---

## Group-Relative Normalization

```python
import torch

def grpo_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """
    rewards: shape [batch_size, K]
    returns: advantages, shape [batch_size, K]
    """
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True)
    return (rewards - mean) / (std + 1e-8)

# Example: rewards = [[1, 0, 1, 0]] → advantages = [[0.866, -0.866, 0.866, -0.866]]
```

---

## PPO vs. DPO vs. GRPO (3-line comparison)

| | PPO | DPO | GRPO |
|---|---|---|---|
| Critic needed | Yes | No | No |
| Data needed | Fresh rollouts | Offline preference pairs | Fresh rollouts |
| Best for | Any | Human preferences | Verifiable rewards |

---

## Decision Rules

- Use DPO when: you have offline preference pairs and no verifier
- Use GRPO when: your reward is a deterministic verifier (SQL execution, unit tests, math check)
- If 70%+ of steps are zero-gradient: shaped reward or adjust prompt difficulty
- K = 8 is a safe default; increase to 16 for hard tasks with low success rates
- If std of group rewards is always 0: your verifier is returning all-same rewards — something is wrong

---

## Numbers to Remember

- K (num_generations): 8 (default in TRL GRPOConfig)
- KL coef β: same as RLHF, ~0.01–0.1
- Clip range ε: 0.2 (same as PPO default)
- Memory: 2× single model (no critic!) — same as DPO
- DeepSeek-R1 group size: varies; 64 in some experiments

---

## Red Flags

- All-zero advantages every step: verifier returning same reward for all K completions; check reward function
- Very high reward variance (std > 1): reward is noisy or poorly calibrated — add clipping
- KL growing past 5 nats: increase β or reduce learning rate
- Zero unique prompts in a batch: batch collation bug — GRPO needs diversity across prompts in the batch
