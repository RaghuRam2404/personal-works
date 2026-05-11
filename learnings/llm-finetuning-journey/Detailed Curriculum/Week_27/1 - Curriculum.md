# Week 27 — Phase 3 Gate

## What This Week Is

Phase 3 Gate is not a new topic. It is a structured self-assessment to determine whether you have absorbed the key skills from Weeks 17–26 and are ready to enter Phase 4 (The Fine-Tuning Stack: SFT, LoRA, QLoRA).

**Do not skip this week.** Skipping gates is how engineers end up doing Phase 6 work with Phase 2 knowledge gaps. The gate is for you — it has no external grader.

## Learning Objectives

By the end of this week, you will be able to:

- Self-certify that each Phase 3 gate criterion is met with evidence
- Identify any areas that need review before Phase 4
- Produce a concise Phase 3 retrospective
- Articulate your Phase 4 plan with realistic estimates

---

## Gate Criteria

Read each criterion. For each one, you must have specific evidence — a file path, a W&B link, a code snippet, or a demonstrated result. "I think I understand it" is not evidence.

**Criterion 1: You have trained a 50M-parameter LM from scratch.**
- Evidence: HuggingFace Hub link to your checkpoint OR local checkpoint path
- Evidence: W&B run URL showing val loss decreasing over at least 15,000 steps
- Evidence: `count_params(model)` output showing 50–65M parameters
- Minimum bar: val loss < 4.0 at some checkpoint

**Criterion 2: You can compute perplexity yourself.**
- Evidence: `eval.py` from Week 22 that computes perplexity correctly
- Evidence: `compute_perplexity(model, val_path)` returns a number between 15 and 60
- Evidence: You can explain the formula (Perplexity = exp(mean CE loss)) and why averaging loss before exponentiation is correct

**Criterion 3: You have a v1 5K-example domain dataset.**
- Evidence: HuggingFace Hub dataset URL (private or public)
- Evidence: `len(train_split) >= 4000` and `len(val_split) >= 1000`
- Evidence: All examples have the ChatML format with schema in user message and SQL-only in assistant message
- Evidence: ≥ 98% of SQL examples pass `sqlglot.parse(dialect="postgres")`

**Criterion 4: You can read any modern LLM technical report and identify architectural and training choices.**
- Evidence: Your `week-24-sota-comparison.md` filled with real numbers for all 5 models
- Evidence: You can explain (without looking it up) what GQA is, what MoE is, what FIM is
- Self-test: Cover your notes and answer — "What is the architecture of Qwen2.5-Coder-7B and why should I fine-tune it for SQL?"

---

## Self-Assessment Process

Work through each criterion systematically. For each, write your evidence in `phase3-gate.md`:

```markdown
# Phase 3 Gate Assessment

**Assessment date:** [today's date]

## Criterion 1: 50M LM Trained From Scratch
- Checkpoint: [link or path]
- W&B run: [link]
- Parameter count: [N]M
- Best val loss: [X.XX]
- Status: PASS / CONDITIONAL / FAIL

## Criterion 2: Can Compute Perplexity
- eval.py location: [path]
- Perplexity on FineWeb-Edu val: [X.X]
- Can explain the formula: YES / NO
- Status: PASS / CONDITIONAL / FAIL

## Criterion 3: v1 5K Domain Dataset
- HuggingFace Hub URL: [link]
- Train examples: [N]
- Val examples: [N]
- SQL validity rate: [N]%
- Status: PASS / CONDITIONAL / FAIL

## Criterion 4: Can Read LLM Technical Reports
- Sota comparison doc: [path]
- Self-test: Explain Qwen2.5-Coder-7B architecture: [your answer]
- Status: PASS / CONDITIONAL / FAIL

## Overall Decision
[READY FOR PHASE 4 / NEEDS REMEDIATION / NOT READY]

## Remediation Plan (if needed)
[What specifically will you fix and by when]
```

### Status Definitions

**PASS:** All evidence is present and meets the minimum bar.

**CONDITIONAL:** Evidence is partial (e.g., val loss is 4.5 instead of 4.0, or dataset has 4,200 examples instead of 5,000). You may proceed to Phase 4 but must complete the remaining work within the first 2 weeks of Phase 4.

**FAIL:** Core deliverable is missing (no trained model, no dataset, no evaluation code). You must spend 1 more week on the failed criterion before proceeding.

---

## What "Needing Remediation" Looks Like

**Common Phase 3 gaps:**

1. **Pretraining run did not complete:** Colab sessions disconnected, or you spent more than $20 on compute. Remediation: run a shorter training (500M tokens instead of 2B) to at least have a functional model.

2. **Dataset has only 2,000 examples:** Remediation: complete Tier 3 self-instruct generation for 1 more week. A 3,000-example dataset is acceptable to pass CONDITIONAL.

3. **Cannot explain Chinchilla without notes:** Remediation: redo the Week 17 exercises. Rewrite the formula by hand. Apply it to 3 compute budgets. Do not look at your notes.

4. **eval.py gives wrong perplexity (too high or too low):** Remediation: cross-check against `lm-evaluation-harness` perplexity on the same data. If they disagree by > 50%, debug your eval.py.

---

## Phase 4 Preview

If you pass the gate, here is what Phase 4 will teach you:
- Week 28: What fine-tuning actually is (continued pretraining vs. SFT vs. instruction tuning)
- Week 29: Full SFT of Qwen2.5-0.5B with HuggingFace `SFTTrainer`
- Week 30: LoRA — the math and the implementation
- Week 31: QLoRA — 4-bit quantization for 7B models on a single A100

Your domain dataset (postgres-sql-v1) will be used starting in Week 29.

---

## Connections

This gate closes Phase 3 (Weeks 17–26), which covered pretraining mechanics, scaling laws, evaluation, and dataset curation for your domain. The four gate criteria directly reflect Phase 3's core deliverables: a trained 50M-parameter LM from scratch (Weeks 17–20), a working perplexity evaluator (Week 22), a 5K-example domain dataset in ChatML format (Weeks 23–25), and the ability to read and compare modern LLM technical reports (Week 24). Each criterion maps to work you should have already done; this week exists to make that assessment explicit and evidence-backed.

Phase 4 (Weeks 28–40) starts immediately after and assumes all four criteria are met. Fine-tuning with LoRA and QLoRA on your domain dataset (Week 29 onward) requires the dataset built in Phase 3. The supervised fine-tuning runs in Phase 4 use the Llama-style architecture you analyzed in Week 24. If you skip the gate or pass it conditionally on dataset size, you will hit this gap in Week 29 when `SFTTrainer` requires a minimum of 4,000 examples to avoid severe overfitting.

## Common Misconceptions / Pitfalls

- **Bluffing the gate.** Reading your own old code and going "yeah I get this" is not the same as being able to re-derive Chinchilla scaling on a blank page or explain why your val perplexity stalled at 30. The gate must be timed and from memory where the criteria say so.
- **Treating the 50M training run as a one-shot success.** If your loss diverged once and you fixed it by lowering LR, that is a pass. If it never diverged, you may have been over-cautious — note this and aim for the edge in Phase 4.
- **Using English perplexity as a proxy for SQL quality.** Perplexity on FineWeb-Edu validates language modeling. It says nothing about whether your dataset is good for SQL fine-tuning. Don't conflate the two metrics during the gate review.
- **Skipping the dataset card.** A 5K dataset without provenance, license info, and known biases is not really a dataset — it's a pile of strings. The gate requires real documentation.
- **Confusing "I read the paper" with "I can teach the paper."** The Llama 3 / Qwen2.5 / DeepSeek-V3 reading week (Week 24) outcomes need to be writeable on a whiteboard in 5 minutes. If you cannot teach a peer the architecture differences, you have not actually read the paper.

## Time Allocation (6–8 hrs)

- 2h: Work through each gate criterion honestly, collect evidence
- 1h: Write `phase3-gate.md` with all evidence documented
- 1h: Remediation (if any criterion is FAIL or CONDITIONAL)
- 1h: Write Phase 3 retrospective (what did you learn, what would you do differently)
- 1h: Plan Phase 4 schedule for the next 3 months
- 0.5h: Commit all gate documents
