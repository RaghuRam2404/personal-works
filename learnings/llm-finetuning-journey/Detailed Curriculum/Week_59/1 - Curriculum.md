# Week 59 — DPO on a Refreshed Preference Dataset

## Learning Objectives

By the end of this week, you will be able to:

- Build a refreshed, high-quality 5K SQL preference dataset using your v3 SFT model as the policy
- Apply DPO training starting from your SFT-v3 checkpoint
- Tune DPO's beta hyperparameter for the SQL execution domain
- Detect and handle the DPO loss-going-negative problem
- Evaluate whether DPO improved execution accuracy vs. the SFT baseline

## Why Refresh the Preference Dataset?

In Phase 5 (Week 44), you built a preference dataset using your Phase 4 SFT model. That model was weaker. The "rejected" SQL it generated may now be trivially wrong — queries with obvious syntax errors that your current, much stronger v3 model would never produce. Training DPO on trivial rejected-vs-chosen pairs produces negligible improvement.

The principle: **DPO preference pairs should be "hard" pairs.** Hard means: both chosen and rejected are plausible attempts, but chosen is better in a meaningful, non-trivial way. Easy means: chosen is valid SQL, rejected has a syntax error any 7B model would fix.

For your domain, "hard" preference pairs look like:
- Chosen: uses `time_bucket('1 hour', time)` correctly with proper GROUP BY; Rejected: uses `DATE_TRUNC('hour', time)` which returns the same values but without TimescaleDB chunking benefits (functionally correct but not idiomatic for TimescaleDB)
- Chosen: uses a CTE for clarity on a complex 4-table join; Rejected: uses a deeply nested subquery that's harder to maintain
- Chosen: uses an index-aware query plan (EXPLAIN shows index scan); Rejected: same logic but written to force a full table scan

These pairs teach the model to prefer not just "correct SQL" but "idiomatic, efficient, TimescaleDB-native SQL."

## Concepts

### Building the Refreshed Preference Dataset

**Strategy:** Sample prompts from your v3 dataset. For each, generate:
- **Chosen:** Your best available answer — either the v3 training example (teacher-generated + filtered) or, if your SFT-v3 model gets it right, the SFT model's output
- **Rejected:** Generate 4–8 SQL candidates using your SFT-v3 model with temperature 0.8 (stochastic). Select the one that is most wrong in a meaningful way (not just a syntax error)

**On-policy vs. off-policy preference data:** DPO was derived assuming the reference model's distribution. If your rejected examples come from the same model you are training (SFT-v3 with temperature), the pairs are on-policy and DPO's KL constraint is well-calibrated. Off-policy pairs (from the Phase 5 model) are less efficient.

**Execution-based preference labeling:** The cleanest label for SQL:
- Run all candidates against Postgres
- If chosen executes correctly and matches reference output, mark as chosen
- Among rejected, prefer the one that executes (wrong result) over the one that errors (syntax error) — execution-wrong is "harder" than syntax-error and produces better DPO signal

**Format (TRL DPO format):**
```json
{
  "prompt": "<system prompt + schema + user question>",
  "chosen": "<correct SQL>",
  "rejected": "<incorrect but plausible SQL>"
}
```

### DPO Hyperparameters

The key DPO hyperparameter is **beta** (β), the KL-divergence constraint strength:
- β = 0.1: weak constraint, aggressive preference optimization, risk of distribution collapse
- β = 0.3: moderate (default, works well for most tasks)
- β = 0.5: strong constraint, conservative, safe but may underfit preference signal

For SQL with verifiable labels (execution correctness), you can afford lower beta (0.1–0.2) because the preference labels are objective and clean. For subjective tasks (helpfulness, style), higher beta is safer.

**Other DPO settings:**
- Learning rate: 5e-5 (lower than SFT's 2e-4 — DPO is a more delicate optimization)
- Epochs: 1–2 (DPO overfits faster than SFT; 5K examples is small)
- Batch size: smaller than SFT (DPO trains on pairs, so effective token count per step is 2× higher)

### The DPO Loss and Its Failure Modes

The DPO loss:

```
L_DPO = -log σ(β * (log π_θ(y_c|x)/π_ref(y_c|x) - log π_θ(y_r|x)/π_ref(y_r|x)))
```

This loss is minimized when the model assigns higher relative probability to chosen vs. rejected, relative to the reference model.

**Loss going negative:** If the DPO loss goes below 0 during training, it means the model has completely learned to separate chosen from rejected — but it may have collapsed the distribution (over-concentrated on a few SQL patterns). This is a warning sign. DPO loss consistently at -1.0 or lower without improvement in eval accuracy suggests the model is fitting training pairs but not generalizing.

**Reward margin monitoring:** Log the implicit reward margin: `log π_θ(y_c|x) - log π_θ(y_r|x) - (log π_ref(y_c|x) - log π_ref(y_r|x))`. It should be positive and increasing. If it stays at 0, DPO is not learning.

### Reference Model Handling

DPO requires a reference model `π_ref` — the frozen SFT model. The loss computes log-probabilities under both the training model and the reference. TRL's `DPOTrainer` handles this automatically when you pass `model` (training) and `ref_model` (frozen SFT-v3 checkpoint).

With LoRA, you can use the base model + merged adapters as the reference, or simply pass the SFT checkpoint directly. The reference model must be identical to the model at the start of DPO training.

### Common Misconceptions and Pitfalls

**"DPO is just a fancier SFT."** Fundamentally different: SFT maximizes log P(chosen); DPO maximizes log P(chosen) while minimizing log P(rejected), with a KL constraint to the reference. The rejected signal is essential — without it, DPO becomes SFT.

**"5K pairs is enough for DPO to substantially outperform SFT."** 5K well-chosen pairs can yield 3–7pp improvement. But the improvement is not guaranteed — it depends entirely on pair quality. Hard, on-policy pairs with meaningful differences are far more effective than easy pairs.

**"I can reuse the Phase 5 DPO dataset."** You can add it to the pool, but prioritize the freshly generated on-policy pairs. The Phase 5 rejected examples are much weaker than what your v3 SFT model produces.

## Time Allocation (6–8 hrs)

- 1.5h: Build 5K preference pairs (generate candidates with SFT-v3, execute, label)
- 0.5h: Audit 50 random pairs — verify "hard" criterion is met
- 1h: Configure and test DPO training script (100-step smoke test)
- 2.5h: Run DPO on Colab Pro (5K pairs at LoRA DPO fits on A100)
- 0.5h: Evaluate DPO checkpoint vs SFT-v3
- 0.5h: Push checkpoint; commit; log W&B

## Connections

This week builds on Weeks 43–45 (DPO basics, preference dataset construction, and beta hyperparameter tuning from Phase 5) and directly on Week 58 (SFT v3 on the 25K refreshed dataset). The SFT-v3 checkpoint from Week 58 is your starting policy and your reference model for DPO — without a strong SFT foundation, DPO's preference signal has nothing stable to build on. The on-policy preference dataset you build here uses SFT-v3 as the generator, which is what makes these pairs harder and more informative than the Phase 5 pairs built from the weaker v1 model.

Week 60 runs GRPO on top of the DPO checkpoint produced here. The DPO run's beta setting and the reward margin you observe will inform whether GRPO needs to be conservative (high KL penalty) or can be aggressive (low KL). Weeks 61–62 then evaluate the resulting model head-to-head against frontier models; the DPO checkpoint is one of the candidate models in that comparison, so a failed or collapsed DPO run this week directly undermines the evaluation results two weeks later.
