# Week 42 TakeAway — PPO and RLHF

**One-liner:** PPO clips the policy update ratio to prevent overshooting; RLHF adds a KL penalty to prevent reward hacking.

---

## Key Formulas

```
# Probability ratio
r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)

# PPO clipping loss (minimize)
L_CLIP = -E_t [ min( r_t · A_t,  clip(r_t, 1-ε, 1+ε) · A_t ) ]

# GAE advantage
δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
A_t = Σ_{k≥0} (γλ)^k · δ_{t+k}

# RLHF total reward
r(x,y)_t = r_RM(x,y) · 1[t=T] - β · KL(π_θ(·|ctx_t) || π_ref(·|ctx_t))
```

---

## Key Code Pattern (GAE)

```python
def gae(rewards, values, gamma=0.99, lam=0.95):
    T = len(rewards)
    adv = torch.zeros(T)
    gae = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t+1] - values[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
    return adv, adv + values[:T]  # advantages, returns
```

---

## InstructGPT 3-Stage Summary

| Stage | Input | Output | Loss |
|---|---|---|---|
| SFT | demonstrations | fine-tuned LM | cross-entropy |
| RM | preference pairs (w, l) | reward model | -log σ(r_w − r_l) |
| PPO | prompts + RM + ref | aligned LM | PPO clip + KL penalty |

---

## Decision Rules

- If KL > 10 nats: increase β or reduce PPO learning rate immediately
- If RM score is high but human evals are bad: reward hacking — add execution-based reward or increase β
- Use λ = 0.95, γ = 0.99 as GAE defaults; lower λ if value function is inaccurate
- ε (clip range) = 0.1–0.2; start at 0.2, reduce if policy changes too fast
- Run ≤ 4 PPO epochs per batch to prevent over-optimization

---

## Numbers to Remember

- PPO ε (clip range): 0.1–0.2 (InstructGPT used 0.2)
- GAE λ: 0.95 (TRL default)
- KL penalty β: 0.01–0.1 (tune; start at 0.05)
- 4 forward passes per step: actor, critic, RM, reference model
- RLHF cost multiplier vs. SFT: ~4× compute per step, many more steps needed

---

## Red Flags

- KL growing monotonically past 5 nats without plateau: β too small or β not being applied
- RM score at ceiling but generation quality dropping: reward hacking — add hard execution filter
- Value loss diverging: reduce LR on critic; re-initialize value head from actor weights
- `ratio` tensor contains values > 5: PPO clip not working; check `logprobs` are in log-space not raw space
