# Week 51 TakeAway — Final Iteration and Model Selection

**One-liner:** Stop when diminishing returns dominate; pick the model with best semantic accuracy among those meeting the Gate criterion.

---

## Stopping Rules (pick the best checkpoint when ANY triggers)

| Rule | Trigger | Action |
|---|---|---|
| Diminishing returns | 3 experiments with < 1pp improvement each | Stop, pick best |
| Budget exhaustion | RunPod budget remaining < cost of next experiment | Stop |
| Acceptance met | v3 beats v1 on exec acc by ≥ 5pp | Stop (optionally 1 more) |
| Eval overfitting risk | > 4 experiments on same 200-query eval set | Stop, add 50 blind queries |

---

## Model Selection Criteria (in priority order)

1. Execution accuracy (Gate requirement: beats v1)
2. Semantic accuracy (practical quality: correct rows returned)
3. Complex query performance (tier-specific)
4. KL divergence from SFT (generalization proxy)
5. Generation length (inference efficiency)

---

## Final Report Structure

```markdown
# Phase 5 Final Report
## Executive Summary (3 sentences)
## Model Progression Table (5 models × 5 metrics)
## Key Findings (3–5 bullets)
## Residual Gaps (what v3 still cannot do)
## Phase 6 Roadmap
```

---

## Phase 5 Progress Summary Template

| Model | Exec Acc | Δ over prev | Sem Acc | Key driver |
|---|---|---|---|---|
| v1 (SFT) | baseline | — | baseline | Domain SFT |
| v2 (DPO) | +11pp | +11 | +13pp | Syntax error reduction |
| v3 (GRPO) | +2pp | +2 | −2pp | Complex query improvement |
| v3-iter1 | +3pp | +3 | +1pp | Harder training prompts |
| v3-iter2 | 0pp | 0 | +4pp | Reference SQL in reward |

---

## Phase 6 Dataset Priority (from Phase 5 failure analysis)

- 40%+ examples: multi-JOIN queries
- 20%+ examples: CTEs and window functions
- 10%+ examples: TimescaleDB hyperfunctions
- 30% examples: simple queries (maintenance, prevent forgetting)

---

## Red Flags at End of Phase 5

- Best model has lower semantic acc than v2: Phase 5 GRPO caused regression — document it and fix in Phase 6
- No checkpoint beats v1: major training failure — consult SFT quality before starting Phase 6
- Eval set used > 4 times for iteration: add blind 50-query test before Gate
- KL > 10 nats on best model: Phase 6 GRPO may need higher β — note in handoff document
