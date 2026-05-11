# Week 52 — Phase 5 Gate

## Learning Objectives

By the end of this week, you will be able to:

- Pass the Phase 5 Gate by meeting all four criteria
- Derive DPO, PPO, and GRPO mathematically from first principles
- Demonstrate the complete SFT → DPO → GRPO pipeline applied to your SQL domain model
- Articulate the connection between RL theory (Weeks 41–42) and practical alignment methods (Weeks 43–49)
- Identify the specific gaps in your Phase 5 model and map them to Phase 6 work

## What the Phase 5 Gate Tests

The gate is a real test. If you cannot pass it, you must repeat the weak modules before starting Phase 6. Do not advance just because 12 weeks have passed.

The four gate criteria (from the master curriculum):

1. **You can explain DPO, PPO, and GRPO mathematically** — on paper, without looking at notes.
2. **You have applied SFT, DPO, and GRPO to your domain model in sequence** — with evidence (HF Hub models, W&B runs, eval reports).
3. **Your v3 model beats your v1 model on held-out eval** — specifically on execution accuracy.
4. **You have run at least one GRPO run with executable rewards** — your Week 47–48 run counts.

## How to Prepare for Criterion 1: Mathematical Derivations

This is the hardest criterion for most people because it requires understanding, not memorization.

### PPO (Week 42)

You must be able to:
- Write the clipping objective: `L_CLIP = E_t [min(r_t·A_t, clip(r_t, 1-ε, 1+ε)·A_t)]`
- Explain why the clip is a pessimistic lower bound
- Write the full RLHF reward: `r_t = -β·KL_t + r_RM · 1[t=T]`
- Explain what the reference model does and why it is frozen

### DPO (Week 43)

You must be able to derive the DPO loss from scratch:
1. Start with the KL-constrained RL objective
2. Show the closed-form optimal policy
3. Invert to express the reward in terms of the policy
4. Show that Z(x) cancels in the preference difference
5. Arrive at: `L_DPO = -log σ(β·(log_ratio_w - log_ratio_l))`

This derivation takes about 15 minutes on paper. Practice it once before the gate.

### GRPO (Week 46)

You must be able to:
- State the group-relative advantage formula: `A_i = (r_i - mean(r)) / std(r)`
- Explain why this replaces the critic network (the empirical mean is a reliable baseline for deterministic verifiable rewards)
- Write the GRPO loss (PPO clip applied to group-relative advantages + KL penalty)
- Explain the connection to REINFORCE baseline subtraction (Week 41)

## Gate Checklist

Work through this checklist before declaring yourself ready to move to Phase 6:

**Mathematical:**
- [ ] Can derive DPO loss from KL-constrained RL objective (15 min, on paper)
- [ ] Can state the PPO clipping objective and explain the clip's role
- [ ] Can state the GRPO advantage formula and explain why no critic is needed
- [ ] Can explain the log-derivative trick (basis of all policy gradients)

**Empirical:**
- [ ] `postgres-sqlcoder-7b-v1` is accessible on HF Hub
- [ ] `postgres-sqlcoder-7b-v2-dpo` is accessible on HF Hub
- [ ] `postgres-sqlcoder-7b-phase5-best` (or v3-grpo) is accessible on HF Hub
- [ ] W&B runs for Week 45 (DPO) and Week 48 (GRPO) are logged and accessible
- [ ] Phase 5 final eval report exists (`week-51-iteration/phase5_final_report.md`)
- [ ] v3-best execution accuracy > v1 execution accuracy (quantified in the report)

**Understanding:**
- [ ] Can explain why DPO is off-policy and why this limits it for SQL
- [ ] Can explain why GRPO does not need a critic for SQL (deterministic verifiable reward)
- [ ] Can name all 6 alignment methods from Week 49 and give their best use case
- [ ] Can describe one reward hacking pattern you encountered and how you fixed it

**If any box is unchecked:** Return to the corresponding week's TakeAway and Curriculum. Do not advance to Phase 6 until the checklist is complete.

## Concepts: What Phase 5 Was About

Phase 5 taught you that post-training alignment is a layered process:

**Layer 1 — SFT (Phase 4):** Teach the model the domain. This is the foundation. Without good SFT, DPO and GRPO have nothing to build on.

**Layer 2 — DPO:** Use execution-labeled preference pairs to reduce systematic errors (syntax errors, schema hallucinations). Offline, stable, cheap. Works best on errors the SFT model makes consistently.

**Layer 3 — GRPO:** Use online verifiable rewards to improve complex queries that DPO's static dataset cannot cover. On-policy, explores novel SQL patterns, expensive but powerful.

The key insight of 2025: verifiable rewards (SQL execution, code testing, math checking) enable GRPO to outperform DPO on the hardest queries, because GRPO generates fresh evidence at training time rather than relying on pre-collected preference pairs that may not cover the hardest cases.

## Looking Ahead: Phase 6

Phase 6 (Weeks 53–78) will:
1. Scale the dataset from ~5K to 50K examples
2. Run the full SFT → DPO → GRPO pipeline fresh on the larger dataset
3. Quantize the best model for deployment (GGUF, GPTQ, or AWQ)
4. Deploy the model with a text-to-SQL API endpoint
5. Write up the entire project as a technical report/paper

Your Phase 5 model is the prototype. Phase 6 builds the production version.

## What to Do if You Cannot Pass the Gate

If criterion 1 (mathematical derivations) is the problem: spend 2–3 hours reading the DPO paper Appendix A.1 and Week 43's AssignmentSolutions. The derivation is 6 steps; practice each one.

If criterion 3 (v3 not better than v1): your GRPO training failed at a fundamental level. Return to Week 47 and check: (1) does your reward function return non-degenerate rewards? (2) did W&B show mean_reward improving? (3) was v2-dpo actually better than v1 before GRPO started?

If criterion 2 (missing models): push the models to HF Hub. This is a logistics issue, not a learning issue.

## Connections

Closes: All of Phase 5 (Weeks 41–51).

Opens: Phase 6 (Weeks 53–78) — the capstone, quantization, deployment, and paper.

## Common Misconceptions / Pitfalls

- **Overfitting to your eval set.** If you evaluated multiple checkpoints against your held-out benchmark and selected the best one, that benchmark is no longer truly held out — you have implicitly used it for model selection. Phase 6 should use a separate evaluation set, or at minimum a bootstrap-confidence-interval estimate to account for this selection effect.
- **Not having held-out preference data.** If all your DPO and GRPO preference data came from the same generation process and you evaluated on the same distribution, you cannot distinguish generalization from memorization. At the gate, verify that your held-out eval set contains prompt styles and schema patterns not present in your training pairs.
- **Confusing log-prob increases with quality improvements.** A model can achieve higher log-probability on chosen completions (which DPO optimizes) without generating better SQL in practice. The gate criterion is execution accuracy on a held-out set, not DPO reward margin. A rising reward margin paired with flat execution accuracy is a failure mode, not a success.
- **Over-relying on win rates from a single judge model.** If you used GPT-4o as a judge to compare your model's outputs, that judge has its own biases (length preference, formatting preference) and is not a reliable proxy for execution correctness. Win rates from a single judge count as supporting evidence, not primary evidence. Primary evidence is execution accuracy.
- **Declaring gate pass because all W&B runs completed.** The gate tests understanding, not just task completion. Criterion 1 (mathematical derivations) is the most likely to fail: sitting down with a blank sheet of paper and deriving DPO from the KL-constrained RL objective is different from having watched someone else do it.

## Time Allocation (6–8 hours)

- 1 hour: Work through the Gate Checklist. Identify any unchecked boxes.
- 2 hours: For any unchecked boxes, return to the relevant TakeAway and practice.
- 1 hour: Complete the Gate Assignment (the written derivation and model verification).
- 1 hour: Write the Gate reflection document.
- 1–2 hours: (Optional) Compare your best model to GPT-4o on 10 hard queries as a preview of Phase 6 goals.
