# Week 46 Assignment Solutions

## Task 1 — GRPOTrainer Key Annotations

**Location 1: K completions generation**
```python
# In _generate_completions() or similar:
# The model generates K outputs for each prompt in the batch.
# This is the "group" in GRPO — all K completions share the same prompt.
# K is controlled by num_generations (default: 8)
# The outputs are concatenated into one large batch for reward computation.
completions = model.generate(
    prompt_ids,
    num_return_sequences=self.args.num_generations,  # K
    ...
)
```

**Location 2: Reward computation**
```python
# The reward_fn is called once per (prompt, completion) pair.
# For verifiable rewards, this is where you call execute_sql(), run unit tests, etc.
# Returns a list of K scalars: [r_1, r_2, ..., r_K]
rewards = [self.reward_fn(prompt, completion) 
           for prompt, completion in zip(prompts, completions)]
```

**Location 3: Group-relative normalization (the key innovation)**
```python
# Reshape rewards from [batch*K] to [batch, K]
rewards_grouped = rewards.view(batch_size, num_generations)
# Compute mean and std WITHIN each group (not across the entire batch)
mean_rewards = rewards_grouped.mean(dim=1, keepdim=True)  # [batch, 1]
std_rewards = rewards_grouped.std(dim=1, keepdim=True)    # [batch, 1]
# Normalize: advantage_i = (r_i - mean) / std
advantages = (rewards_grouped - mean_rewards) / (std_rewards + 1e-8)
# This is REINFORCE baseline subtraction applied group-wise (Week 41 connection)
```

**Location 4: PPO clip (same as PPOTrainer)**
```python
ratio = (new_logprobs - old_logprobs).exp()  # π_θ / π_θ_old
pg_loss1 = -advantages * ratio
pg_loss2 = -advantages * ratio.clamp(1 - cliprange, 1 + cliprange)
loss = torch.max(pg_loss1, pg_loss2).mean()  # pessimistic bound
```

**Location 5: KL divergence penalty**
```python
# Per-token KL approximation: log π_θ - log π_ref
kl = new_logprobs - ref_logprobs  # shape: [batch*K, seq_len]
loss = loss + kl_coef * kl.mean()  # added to the policy loss
```

**Common gotchas:**
- TRL's GRPO uses `num_generations` for K — the default may be 4 or 8 depending on version
- The advantage normalization is per-group (per-prompt), not per-batch — easy to implement wrong
- If all K completions in a group get the same reward, std=0, and we divide by eps — advantage = 0, no gradient
- The clip range for GRPO is the same ε as PPO (default 0.2 in TRL)

---

## Task 2 — GRPO Advantages Calculation (SQL Example)

For the stretch goal: rewards = [1, 0, 1, 0]:
```python
import torch
rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
mean = rewards.mean()   # 0.5
std = rewards.std()     # 0.5774...
advantages = (rewards - mean) / (std + 1e-8)
# advantages = [0.866, -0.866, 0.866, -0.866]
# (std of [1,0,1,0] is population std ≈ 0.5, not sample std ≈ 0.577 — verify with ddof=0)
```

Note: PyTorch `tensor.std()` uses ddof=1 (sample std) by default. The advantages will be ±0.866 for the [1, 0, 1, 0] case with ddof=1.

---

## Task 3 — Algorithm Comparison Table

| | PPO | DPO | GRPO |
|---|---|---|---|
| Requires critic | Yes | No | No |
| Requires reward model | Yes (RM) | No (uses labels) | No (uses verifier) |
| On/off policy | On-policy | Off-policy | On-policy |
| Best for | Any reward, established pipeline | Human preference data | Verifiable rewards |
| Memory cost | 4× single model | 2× single model | 2× single model |
| Key hyperparams | ε, β, lam, vf_coef | β | ε, β, K (num_generations) |

---

## How to Verify You Did It Right

1. Your annotated file has ≥ 15 lines of comments across the 5 locations.
2. Your explainer uses no copy-pasted text from the DeepSeekMath paper.
3. The concrete SQL example in the explainer shows computed advantages (not just the formula).
4. Can you explain GRPO in 2 minutes to a colleague without looking at your notes?
