# Week 49 TakeAway — Alignment Zoo

**One-liner:** PPO/GRPO = online; DPO/KTO/ORPO/SimPO = offline. Use verifiable rewards → GRPO; human prefs → DPO; unpaired labels → KTO.

---

## Full Comparison Table

| Method | Ref model | Data format | Memory | Best for |
|---|---|---|---|---|
| PPO | Yes | Online rollouts + RM | 4× | Any reward, flexible pipeline |
| DPO | Yes | Paired (w, l) | 2× | Execution/human preferences, offline |
| GRPO | Yes | Online rollouts + verifier | 2× | Verifiable rewards (SQL, code, math) |
| KTO | Yes | Unpaired good/bad | 2× | Production logs, unpaired annotations |
| ORPO | No | Paired (w, l) | 1× | Memory-constrained, 1-stage training |
| SimPO | No | Paired (w, l) | 1× | Pairs with very different lengths |

---

## Key Formulas (1 line each)

```
PPO:   L = -min(r·A, clip(r, 1-ε, 1+ε)·A) - β·KL(π||π_ref)
DPO:   L = -log σ(β·(log_ratio_w - log_ratio_l))
GRPO:  L = PPO_clip(A = group_normalized_reward) + β·KL
KTO:   L = E[f(r_θ(y_u) - r_ref)] - E[f(r_θ(y_d) - r_ref)]   [prospect theory]
ORPO:  L = L_SFT - λ·log σ(log_odds_w - log_odds_l)
SimPO: L = -log σ((avg_logp_w - avg_logp_l) - γ)
```

---

## Decision Tree

```
Have verifiable reward (SQL execution, unit tests)?
  YES → GRPO
  NO → Is data paired?
    YES, with ref model budget? → DPO
    YES, memory constrained? → ORPO or SimPO
    NO (unpaired annotations)? → KTO
    NO (online, any reward)? → PPO
```

---

## TRL Trainer Map

```python
from trl import (
    PPOTrainer,    # kl_coef=0.05
    DPOTrainer,    # beta=0.1
    GRPOTrainer,   # num_generations=8
    KTOTrainer,    # desirable_weight=1.0
    ORPOTrainer,   # lambda_orpo=0.1
    SimPOTrainer,  # gamma=0.5
)
```

---

## Numbers to Remember

- DPO β: 0.1 default
- GRPO K: 8 default
- KTO: works with unpaired (prompt, completion, label) triples
- ORPO, SimPO: no reference model → 1× memory but no KL anchor
- SimPO γ (margin): 0.5 default
- DPO and KTO both need 2× memory (training model + ref model)
