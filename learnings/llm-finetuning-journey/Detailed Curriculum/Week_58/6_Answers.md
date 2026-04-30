# Week 58 Answers

## Q1 — Answer: B

**Why:** The lower starting loss (1.95 vs 2.4) means the CPT checkpoint's next-token predictions on your SFT data are already significantly better than the base model's predictions. CPT has moved the model's weights closer to the target domain distribution. This is the intended effect: CPT pre-fills the model with domain vocabulary and patterns, reducing the amount of learning that SFT needs to do. It typically translates to 1–3 percentage points of final eval improvement from the same number of SFT steps.

**Why A is wrong:** Contamination would cause loss near 0.0, not 1.95.

---

## Q2 — Answer: B

**Why:** A train/val loss gap of 0.42 in SFT is typical and not necessarily alarming. SFT on a diverse 25K dataset does not usually lead to severe overfitting; the validation examples are drawn from the same distribution. The correct diagnostic is to check domain execution accuracy: if the validation loss is slightly higher but execution accuracy is good, the model generalizes well despite the loss gap. Execution accuracy is the ground truth for your use case.

---

## Q3 — Answer: B

**Why:** `DataCollatorForCompletionOnlyLM` masks all tokens that come BEFORE the specified response template. If you give it the user role delimiter instead of the assistant role delimiter, it masks user tokens and computes loss on user question tokens + system prompt tokens (everything from the user delimiter to the end of the conversation). The model learns to predict user questions, not SQL answers — completely wrong direction.

---

## Q4 — Answer: B

**Why:** DPO trains the model on preference pairs to shift output distribution toward better responses. GRPO with executable rewards directly rewards correct SQL at inference time. Each stage typically contributes 5–8 percentage points on domain-specific benchmarks when the data quality is high. The combined effect is typically 10–15 pp over SFT baseline for execution-correct tasks with verifiable reward signals. A final model in the 78–85% range on your custom benchmark is realistic — and may exceed GPT-4o on your specific TimescaleDB domain.

---

## Q5 — Model Answer

The periodic spikes are almost certainly caused by batches containing longer, harder examples (e.g., multi-turn conversations or Expert-difficulty queries) being sampled together. Loss is higher on these examples; when multiple hard examples land in the same batch, the averaged loss spikes. The immediate recovery shows the model returns to its learned state quickly — these hard batches don't destabilize training.

This is not a problem. It is normal behavior in any diverse dataset. If you want to reduce the spikes: increase effective batch size (from 32 to 64), which averages out difficult batches more smoothly. Or implement curriculum training: sort by difficulty and train easy examples first, hard examples later. But for most purposes, the current behavior is fine.

---

## Q6 — Model Answer

Do not continue to epoch 3 based on this data alone.

The accuracy decrease from 71% to 70% at epoch 2 is within measurement noise (the eval uses 200 examples: 1 example difference = 0.5pp). You need to re-run the eval 3 times and report mean ± std to know if this is a real decrease or noise.

Information needed to decide:
- Is the 1pp decrease reproducible across 3 eval runs?
- What is the training loss at epoch 2 vs epoch 1? Is it still decreasing?
- What is the validation loss? Is it still decreasing or has it turned up?

If the validation loss has clearly turned up and the eval accuracy is flat or declining: stop at epoch 2. If the validation loss is still decreasing and the accuracy difference is noise: continue to epoch 3 with early stopping patience of 200 steps on validation loss.

---

## Q7 — Model Answer

Three actions without re-running full SFT:

1. **Targeted LoRA fine-tuning on TimescaleDB subset.** Take the SFT checkpoint and run a short (200–300 step) LoRA fine-tuning run using ONLY the TimescaleDB-specific examples from v3 (your ~1,500–3,000 TimescaleDB examples). Use a low learning rate (5e-5) to avoid catastrophic forgetting of general SQL. This is a "skill patch" that directly addresses the gap.

2. **Add TimescaleDB examples to the DPO preference dataset (Week 59).** When you build preference pairs for DPO, over-sample TimescaleDB queries. Even if the chosen/rejected distinction is subtle, DPO will push the model's distribution toward better TimescaleDB output. This is Week 59's work but you should plan it now.

3. **Verify that TimescaleDB examples in v3 use the correct function signatures.** Run a manual review of 50 random TimescaleDB examples in your training data. If even 20% have slightly wrong syntax (e.g., wrong argument order for `time_bucket_gapfill`), those examples are actively teaching the model wrong patterns. Fix those before DPO/GRPO.

---

## Q8 — Model Answer

Diagnosing the 12-point gap:

Data coverage: GPT-4o was trained on far more SQL content from more diverse sources. The 12-point gap likely concentrates in: (a) queries requiring implicit schema reasoning (knowing that `time` column is `timestamptz` without being told explicitly), (b) complex join chains with 4+ tables, (c) edge cases in TimescaleDB syntax (hyperfunctions with multiple optional arguments). Run a per-query-type analysis: if the gap is 5% on basic queries and 25% on TimescaleDB-specific queries, it points to data coverage, not model capacity.

Model capacity: 7B vs GPT-4o (estimated 200B+). Pure capacity alone probably accounts for 5–8pp of the gap. You cannot close the full capacity gap, but you can close the domain gap.

Training procedure: GPT-4o has multi-stage RLHF that aligns it to follow complex instructions precisely. Your SFT-only checkpoint has no alignment beyond teacher examples. DPO and GRPO (Weeks 59–60) will directly address this.

Evaluation methodology: Is GPT-4o using the same schema DDL in its prompt that your model uses? If GPT-4o has access to richer schema context (e.g., sample data, column descriptions), the comparison is unfair. Verify that both models receive identical prompts.

Three interventions (ranked):
1. DPO + GRPO training (Weeks 59–60): most impactful, estimated +7–10pp. GRPO with executable rewards directly rewards correct SQL — the most aligned training signal possible.
2. Add 2,000 more Expert-difficulty TimescaleDB examples: targeted data fix for the skill gap, estimated +2–4pp on TimescaleDB queries.
3. Enrich schema context in inference prompts: add column types and sample values to schema DDL at inference time (not training time). GPT-4o benefits from this information; make sure your model also receives it. Estimated +1–3pp at zero training cost.
