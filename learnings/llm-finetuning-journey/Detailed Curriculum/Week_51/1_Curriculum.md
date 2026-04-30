# Week 51 — Iteration Week 2: Pick Your Best Model

## Learning Objectives

By the end of this week, you will be able to:

- Continue the targeted iteration process from Week 50 on the second-priority failure mode
- Run a final comparative evaluation across all models (v1, v2, v3, v3-iter1, and any new checkpoints)
- Select the best model checkpoint based on a clear, multi-metric decision framework
- Freeze the model that will be presented at the Phase 5 Gate (Week 52)
- Produce a final Phase 5 eval report that tells the complete story of v1→v2→v3

## Continuing the Iteration Loop

Week 51 picks up where Week 50 left off. You have:
- v3: your original GRPO model from Week 48
- v3-iter1: the result of Week 50's targeted experiment

This week you address the second failure mode from your Week 50 diagnosis (the one you ranked as medium priority). You will also decide, based on all the evidence, which checkpoint is your "best model."

## Concepts

### When to Stop Iterating

Iteration without a stopping criterion leads to overfitting to the eval set. The stopping rules:

**Rule 1: Diminishing returns.** If three consecutive experiments each improve the target metric by less than 1pp, stop. The marginal return does not justify the RunPod cost.

**Rule 2: Budget exhaustion.** Phase 5 had $60 compute budget. Weeks 47–48 used ~$10–15. Week 50 used ~$6–8. Week 51 has ~$15–25 remaining. Once you exhaust this budget, stop and pick the best checkpoint.

**Rule 3: Acceptance criteria met.** The Phase 5 Gate requires:
- v3 beats v1 on execution accuracy: if this is met, you have passed the main criterion.
- SFT → DPO → GRPO pipeline complete: this is structural, not metric-based.
- Mathematical understanding: this is your job to prepare, not a model metric.

**Rule 4: Overfitting to eval.** If you run more than 4 experiments targeting the same 200-query eval set, you risk unconsciously constructing experiments that overfit to those specific queries. Freeze the eval set and report honestly on it.

### Model Selection Framework

When choosing the best model among (v3, v3-iter1, v3-iter2, ...), use this framework:

**Criterion 1: Execution accuracy on full test set.** The primary metric. If multiple models tie, move to Criterion 2.

**Criterion 2: Semantic accuracy.** More informative than execution accuracy (a model that executes correctly AND returns right rows is better than one that executes but returns garbage).

**Criterion 3: Complex query tier.** Sub-domain performance on the hardest queries. If your deployment focus is complex analytics SQL (TimescaleDB, CTEs, window functions), weight this heavily.

**Criterion 4: KL divergence from SFT.** Lower KL means the model is closer to the SFT reference, which generally means better generalization on out-of-distribution prompts. If two models have nearly identical eval metrics, choose the one with lower KL.

**Criterion 5: Generation length.** Slightly shorter is generally better (fewer tokens = faster inference). If a model produces much longer outputs with the same accuracy, prefer the shorter one.

### Final Eval Report Structure

Your Phase 5 final eval report should tell the complete story:

```markdown
# Phase 5 Final Evaluation Report

## Model Progression
| Model | Config | Exec Acc | Sem Acc | Complex Exec Acc |
|---|---|---|---|---|
| v1 (SFT only) | Week 34 | 68% | 44% | 40% |
| v2 (SFT+DPO) | Week 45 | 79% | 57% | 48% |
| v3 (SFT+DPO+GRPO) | Week 48 | 81% | 55% | 49% |
| v3-iter1 | Week 50 | 84% | 56% | 61% |
| v3-iter2 | Week 51 | 84% | 60% | 62% |

## Selected Model: v3-iter2

**Rationale:** v3-iter2 achieves the highest semantic accuracy (60%) and ties v3-iter1 on 
execution accuracy (84%). It addresses both identified failure modes from Week 50.

## Key Findings
1. DPO reduced syntax errors by 15pp over SFT; GRPO did not further reduce syntax errors 
   but improved complex query handling.
2. Reward function quality (having reference SQL for semantic verification) was the 
   biggest driver of semantic accuracy improvement.
3. Mode collapse (reward_std → 0) was the primary GRPO failure mode; fixed with harder prompts.
```

### What the Phase 5 Gate Will Test

Week 52 is the gate. Based on the master curriculum:

1. Can you explain DPO, PPO, and GRPO mathematically? (Written derivation, not just the recipe)
2. Have you applied SFT, DPO, and GRPO to your domain model in sequence?
3. Does your v3 model beat v1 on held-out eval?
4. Have you run at least one GRPO run with executable rewards?

Prepare for the mathematical derivations:
- REINFORCE policy gradient (Week 41)
- PPO clipping objective with GAE (Week 42)
- DPO loss derivation from KL-constrained RL objective (Week 43)
- GRPO group-relative advantage formula (Week 46)

## The Residual Gap: What GRPO Did Not Fix

Be honest about what v3 does NOT do well. The typical residual gaps at this stage:

1. **Very long queries (5+ JOINs):** Even v3-iter may fail here. Phase 6's larger dataset will address this.
2. **Novel TimescaleDB hyperfunctions:** If not in training data, v3 has not seen them.
3. **Ambiguous prompts:** "Show me recent activity" — what does "recent" mean? v3 may hallucinate a definition.

Document these in the eval report. They become the roadmap for Phase 6.

## Connections

This week concludes: Phase 5 (Weeks 41–51).

Week 52: Gate. You will need your best model, the complete eval report, and preparation for the mathematical derivations.

Phase 6 (Weeks 53–78): Builds directly on v3. The Phase 6 starting point is your best model from this week.

## Common Misconceptions

- "More iteration is always better." After 3–4 targeted experiments, diminishing returns dominate. The remaining gains come from better data (Phase 6), not more tuning.
- "I should pick the model with highest training reward." Pick the model with highest held-out eval metrics. Training reward can be hacked; held-out eval cannot (assuming you did not use it to tune hyperparameters).
- "v3 must beat v2 on ALL metrics." The gate only requires v3 to beat v1 on execution accuracy. If v3 is slightly worse than v2 on semantic accuracy but better on everything else, that is a reasonable tradeoff to document.

## Time Allocation (6–8 hours)

- 30 min: Review Week 50 results. Decide what Week 51 experiment addresses.
- 30 min: Write the Week 51 hypothesis and experiment plan.
- 3–4 hours async: Run the second targeted GRPO experiment.
- 1 hour: Run final evaluation on all models (v1, v2, v3, v3-iter1, v3-iter2).
- 1 hour: Write the final Phase 5 eval report.
- 30 min: Select the best model, push it to HF Hub with a clear version tag.
