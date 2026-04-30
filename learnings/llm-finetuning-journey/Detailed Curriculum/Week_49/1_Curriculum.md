# Week 49 — KTO, ORPO, and the Alignment Zoo

## Learning Objectives

By the end of this week, you will be able to:

- Describe KTO, ORPO, and SimPO at a conceptual level and identify what problem each solves
- Produce a comparison table of all alignment methods covered in Phase 5 (PPO, DPO, GRPO, KTO, ORPO, SimPO)
- Articulate for each method: the data requirement, the loss formulation at a high level, and the ideal use case
- Explain why you chose GRPO for your SQL domain (and when you would choose each alternative)
- Use this week to consolidate Phase 5 knowledge before the Gate (Week 52)

## Why a Survey Week?

The alignment landscape is moving rapidly. Between 2022 and 2025, at least a dozen preference optimization methods were published. You do not need to implement all of them — but you need to know they exist, understand their core tradeoffs, and be able to answer "why did you use DPO instead of KTO?" in a technical interview or paper discussion.

This is a skim-level reading week. The goal is breadth, not depth.

## Method Overviews

### KTO: Kahneman-Tversky Optimization

**Paper:** [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306) — Ethayarajh et al. 2024.

**Core insight:** DPO requires paired preferences (chosen + rejected for the same prompt). KTO does not. KTO works with a simpler dataset: a flat list of (prompt, completion, label) triples where label ∈ {desirable, undesirable}. No pairing required.

**Why this matters:** Many real datasets have unpaired annotations — you have many "good" SQL examples and many "bad" SQL examples, but they are not paired (the good and bad SQL are for different prompts). DPO cannot use this directly; KTO can.

**Mathematical basis:** KTO draws from Kahneman-Tversky prospect theory — the psychological finding that humans are more sensitive to losses than gains. The loss function is:

```
L_KTO(π_θ; π_ref) = E[λ_u · f(r_ref − r_θ(y_u|x))]   for undesirable
                  − E[λ_d · f(r_θ(y_d|x) − r_ref)]     for desirable
```

where r_θ(y|x) = β · log(π_θ(y|x)/π_ref(y|x)) is the implicit reward, and r_ref is a reference point (the KL-adjusted reward of the reference model's output), and f is a sigmoid.

**When to use KTO:**
- When you have unpaired good/bad examples (not preference pairs)
- When your "desirable" data is much more common than "rejected" data
- When you want a simpler data pipeline than DPO

**Limitations for SQL:** KTO still requires log-ratio computation with a reference model (same as DPO). For verifiable SQL rewards, GRPO is still better because it uses execution-time feedback.

### ORPO: Monolithic Preference Optimization without Reference Model

**Paper:** [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691) — Hong et al. 2024.

**Core insight:** DPO requires a frozen reference model (π_ref) for every training step (2 forward passes). ORPO eliminates the reference model entirely by incorporating the preference signal directly into the SFT cross-entropy loss as an odds ratio penalty.

**Loss:**
```
L_ORPO = L_SFT - λ · log σ(log(odds_w) − log(odds_l))
where odds_y = π_θ(y|x) / (1 − π_θ(y|x))  [per-token]
```

The key: ORPO trains the model to both generate good responses (via SFT loss on chosen) and reject bad responses (via the odds ratio penalty on rejected), in a single training step with one model.

**Benefits:**
- 1× memory instead of 2× (no reference model)
- Faster training (one forward pass, not two)
- Can be applied during SFT (not as a separate stage)

**Limitations:** Without a reference model, the KL regularization is implicit and weaker. The model can drift from the pretrained distribution more easily. Works best when chosen examples are clearly better than rejected.

**For SQL:** ORPO is attractive if memory is very constrained (e.g., Colab Free). But without a reference model anchor, the SQL schema adherence may degrade faster than with DPO.

### SimPO: Simple Preference Optimization

**Paper:** [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734) — Meng et al. 2024.

**Core insight:** DPO's reference model normalization is expensive and may not be necessary. SimPO uses average log-probability (normalized by sequence length) as the reward, with a target margin γ:

```
L_SimPO = -log σ(
    (1/|y_w|)·log π_θ(y_w|x) − (1/|y_l|)·log π_θ(y_l|x) − γ
)
```

No reference model. The length normalization (dividing by |y|) prevents the model from preferring shorter completions for the wrong reasons (long completions naturally have lower cumulative log-probability). The margin γ acts as a threshold — the model must prefer chosen over rejected by at least γ log-prob units.

**Benefits:**
- No reference model (cheaper than DPO)
- Length normalization is built in (addresses a known DPO failure mode)
- γ gives direct control over the minimum preference gap

**Limitations:** No KL regularization. The model can drift from the pretrained distribution. The optimal γ is dataset-dependent and requires tuning.

**For SQL:** SimPO's length normalization is useful if your SQL preference pairs have very different lengths. The lack of reference model is a drawback if you need strong regularization to preserve schema adherence.

### The Zoo in Summary

| Method | Ref model | Data format | Key innovation | Best for |
|---|---|---|---|---|
| PPO | Yes | Online rollouts + RM | Clipped policy gradient | Any reward, flexible |
| DPO | Yes | Paired preferences | Closed-form RL solution | Human preferences, offline |
| GRPO | Yes | Online rollouts + verifier | No critic, group normalization | Verifiable rewards |
| KTO | Yes | Unpaired good/bad | Prospect theory loss | Unpaired annotation datasets |
| ORPO | No | Paired preferences | Monolithic SFT+preference | Memory-constrained, 1-stage |
| SimPO | No | Paired preferences | Length normalization + margin | No-ref-model with length control |

### Which Would You Use for SQL?

For your specific problem (SQL execution, verifiable reward, PostgreSQL):

1. **GRPO** (what you used in Weeks 47–48): The correct choice for verifiable rewards. Online generation + execution = fresh, accurate reward signal. No reward model or human annotation needed.

2. **DPO** (what you used in Week 45): Good for the initial preference dataset (Week 44) where you have execution-based labels. Offline, stable, easy to implement.

3. **KTO**: Would be useful if you had a large bank of labeled SQL (good vs. bad) from production logs without pairing. Not better than GRPO for your current setup.

4. **ORPO**: Would save memory on Colab Free, but the weaker regularization risks schema hallucination. Not recommended over DPO for your domain.

5. **SimPO**: Potentially useful as a DPO replacement if length normalization helps (complex queries tend to be longer; DPO can penalize long correct SQL). Worth experimenting in Week 50 if DPO reward_margin is low on complex queries.

## Connections

Consolidates: All of Phase 5 (Weeks 41–48).

Week 52 (Gate): You will need to explain all methods at the gate. The comparison table you build this week is the preparation.

## Common Misconceptions / Pitfalls

- **Confusing the loss objectives.** KTO, ORPO, and SimPO each optimize a fundamentally different objective: KTO uses a prospect-theory reference point, ORPO uses an odds-ratio penalty baked into SFT, and SimPO uses length-normalized log-probabilities with a margin. Mixing up which method needs a reference model and which does not is the most common error when reading papers or configuring TRL trainers.
- **Assuming one method always beats another.** None of these methods dominates universally. AWQ is usually better than GPTQ on activation-outlier models, but GPTQ sometimes wins in practice. Similarly, SimPO's length normalization helps on long-completion tasks but can hurt on short ones. Always run ablations on your own domain before concluding anything.
- **Skipping ablations.** With 6 alignment methods now in your toolkit, it is tempting to pick one and call it done. For SQL with verifiable rewards, the right answer is empirical: run DPO, run KTO, measure execution accuracy, compare. Arguing from theory alone is insufficient.
- **Treating KTO, ORPO, or SimPO as drop-in DPO replacements without re-tuning hyperparameters.** Each method has different sensitivity to its key hyperparameter (KTO's beta, ORPO's lambda, SimPO's gamma). The defaults in TRL were tuned on general chat tasks. For SQL execution, you will need to sweep at least 3 values of the primary hyperparameter before comparing methods fairly.
- **Using the same preference dataset across methods.** KTO works with unpaired labels; DPO and SimPO require paired preferences. If you built your preference dataset for DPO (paired), it is not directly reusable for KTO without reformatting. Verify dataset format compatibility before switching methods.

## Time Allocation (6–8 hours)

- 1.5 hours: Skim KTO paper (abstract, Section 2, Table 1 in the paper).
- 1 hour: Skim ORPO paper (abstract, Section 3 on the loss, experiments).
- 1 hour: Skim SimPO paper (abstract, Section 3, compare to DPO results table).
- 1 hour: Build the comparison table (Assignment Task 1).
- 1 hour: Write the "For my SQL domain" analysis (Assignment Task 2).
- 1–2 hours: TRL docs browsing — find KTOTrainer, ORPOTrainer, SimPOTrainer. Note the config differences.
