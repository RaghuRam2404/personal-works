# Week 58 — Full SFT on the 50K v3 Dataset

## Learning Objectives

By the end of this week, you will be able to:

- Configure a full SFT run on your v3 dataset starting from the CPT checkpoint
- Choose the right LoRA rank, learning rate, and sequence length for a 25K+ example dataset
- Monitor SFT for overfitting, convergence, and domain coverage throughout training
- Evaluate the SFT checkpoint against your Phase 5 baseline to verify improvement
- Save and push the SFT checkpoint in the correct format for downstream DPO training

## The Central SFT Run

This is the single most important training step of the entire 18-month course. Everything from Weeks 53–57 — the quality data strategy, synthetic generation, filtering, multi-turn additions, CPT — feeds into this one run. Your v3 SFT model will become the reference point for all subsequent evaluations.

You have done SFT before (Phase 4, Week 35; Phase 5, Week 42). This run differs in three ways:
1. Far more data: 25K+ examples vs. 5–10K in prior phases
2. Better initialization: CPT checkpoint from Week 57, not the base model
3. Higher quality data: filtered through LLM-as-judge, diverse schemas, multi-turn

## Concepts

### Hyperparameter Selection for Large-Scale SFT

At 25K+ examples with a 7B model and LoRA, the key decisions:

**LoRA rank:** 32–64. At Phase 4, you likely used rank 16. With more data and a more capable initialization (CPT), you can afford higher rank. Higher rank = more parameters trained = better capacity, but also more memory and more risk of overfitting on less common skills. Rank 32 is a safe default; rank 64 gives 1–2% extra performance at 4× the LoRA parameter count.

**Learning rate:** 2e-4 with cosine decay. This is the established default for LoRA SFT on 7B models. Do not go higher — CPT already shifted the model toward your domain, so large LR updates risk undoing the CPT gains. Consider 1e-4 if you see training instability.

**Batch size:** Effective batch size 32–64. With gradient accumulation: per-device batch 4 × accumulation 8 = 32 effective. At H100 80GB with LoRA rank 32 and sequence length 2048: per-device batch 8 is achievable.

**Epochs:** 2–3. Unlike CPT (1 epoch strictly), SFT can and should run for 2–3 epochs. At 25K examples, 2 epochs gives 50K gradient updates at effective batch 32. Monitor validation loss — if it starts increasing, stop early.

**Sequence length:** 2048. Your multi-turn examples are designed to fit within 2048 tokens. If the p99 token length is > 1800, increase to 4096 (but this quadruples attention memory cost).

**Warmup steps:** 50–100 steps (0.1–0.5% of total). Less warmup than CPT because the model is already well-initialized.

### The SFT Loss and What It Tells You

SFT cross-entropy loss at initialization (starting from CPT checkpoint on SQL pairs) should be around 1.2–2.0, depending on how much SQL overlap there is between CPT corpus and v3 training set. If starting loss is below 1.0, your data is too similar to CPT corpus (potential contamination). If above 2.5, the CPT checkpoint hasn't provided much initialization advantage.

During training, watch:
- **Training loss:** Should decrease smoothly. If it spikes, check for bad data batches.
- **Validation loss:** Should decrease for ~1.5 epochs then plateau or slightly increase. Stop before it clearly increases.
- **Domain benchmark accuracy:** Run your 200-example eval every 500 steps. This is the metric you actually care about.

### Evaluation During Training

Do not rely on training loss alone. Run your quick eval script every 500–1000 steps:

```python
# Sample 50 examples from your custom eval set
# Generate SQL with greedy decoding
# Execute against Postgres
# Report execution accuracy
```

This gives you the actual metric that matters for your use case. Training loss is a proxy; execution accuracy is the truth.

### The Prompt Template

Your v3 dataset is stored in chat format (messages list). You need to apply the Qwen2.5 chat template to convert messages to a flat string with the correct special tokens. Use:

```python
tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False  # False for training; True for inference
)
```

Verify the formatted string looks correct for 3 examples before training. A common error is the template producing malformed output for multi-turn examples.

### Overfitting Signals at Scale

With 25K examples over 2–3 epochs, overfitting is less severe than in Phase 4 (small dataset). But watch for:

- **Schema memorization:** The model correctly answers questions from schemas in training but fails on held-out schemas. Test with 5 schema structures not in v3.
- **Phrasing rigidity:** The model only works when the question uses exact phrasings from training data. Test with paraphrased versions of training questions.
- **TimescaleDB concentration:** If TimescaleDB examples are underrepresented, the model may improve generally but regress specifically on your target domain. Track TimescaleDB eval separately.

### Common Misconceptions and Pitfalls

**"Higher LoRA rank is always better."** At rank 64 with 25K examples, you are training 160M+ LoRA parameters. This can overfit if your data diversity is insufficient. Test rank 32 vs 64 with 500-step pilots before committing.

**"I should train until validation loss hits a hard floor."** Validation loss can plateau while your actual eval metric (execution accuracy) continues improving. Use execution accuracy as the stopping criterion, not validation loss.

**"The SFT checkpoint is the final model."** No — DPO (Week 59) and GRPO (Week 60) will further improve it. Treat this checkpoint as a strong starting point, not the end.

## Connections

This week's output (`qwen2.5-coder-7b-postgres-sft-v3`) is:
- The reference model for DPO (Week 59) — both the SFT reference and the starting point
- The baseline to beat for GRPO (Week 60)
- The "SFT-only" condition in your evaluation ablation table (Week 69)

## Time Allocation (6–8 hrs)

- 1h: Final data validation — run the full pipeline end-to-end on 10 examples before training
- 0.5h: Configure training script (hyperparameters, logging, checkpoint schedule)
- 0.5h: Smoke test locally (100 steps on 1K examples, verify loss decreases)
- 0.5h: Spin up RunPod H100, upload checkpoint and dataset
- 4h: Run full SFT + monitor (3–4 hours active training)
- 0.5h: Evaluate SFT checkpoint; compare to Phase 5 baseline
- 0.5h: Push to HuggingFace; terminate RunPod; commit code
