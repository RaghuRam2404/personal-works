# Week 58 Assignment — Full SFT on v3 Dataset

## Setup Checklist

- [ ] v3 dataset on HuggingFace (`<your-handle>/postgres-sql-v3`) accessible
- [ ] CPT checkpoint (`<your-handle>/qwen2.5-coder-7b-postgres-cpt`) accessible
- [ ] RunPod H100 access; budget ~$15–20 for this run
- [ ] Your 200-example custom eval set (PostgreSQL/TimescaleDB benchmark) ready
- [ ] W&B project `week-58-sft` created

---

## Task 1 — Final Data Validation

**Goal:** Catch any format errors before wasting GPU time.

**Requirements:**
Run `validate_v3.py` that:
- Loads every example in `v3_final.jsonl`
- Applies Qwen2.5-Coder chat template to each
- Checks: no example exceeds 2048 tokens after tokenization
- Checks: all examples have at least one assistant turn
- Checks: no empty assistant turns (empty SQL)
- Checks: total token count (should be 40–70M tokens for 25K+ examples at ~1600 tokens average)
- Reports any violations; remove violating examples and save cleaned dataset

**Acceptance criteria:** 0 validation errors in the final dataset.

**Deliverable:** `validate_v3.py` + final cleaned `v3_train_final.jsonl` committed.

---

## Task 2 — SFT Training Script

**Goal:** Write the full SFT training script with proper logging and checkpointing.

**Requirements:**
Write `train_sft_v3.py` using Unsloth + TRL SFTTrainer:

```python
# Key configuration
model, tokenizer = FastLanguageModel.from_pretrained(
    "<your-handle>/qwen2.5-coder-7b-postgres-cpt",  # start from CPT
    max_seq_length=2048,
    load_in_4bit=False,  # bf16 on H100
)

peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    lora_dropout=0.0,
    bias="none",
)

training_args = SFTConfig(
    output_dir="./sft_v3_output",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,   # effective batch = 32
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    bf16=True,
    logging_steps=25,
    eval_steps=500,
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="wandb",
    run_name="week-58-sft-v3",
    max_seq_length=2048,
)
```

The trainer must use `DataCollatorForCompletionOnlyLM` to compute loss only on assistant tokens.

**Deliverable:** `train_sft_v3.py` committed and runs to completion.

---

## Task 3 — Run SFT on RunPod H100

**Requirements:**
- Spin up H100 80GB; upload script and dataset
- Run training; monitor W&B for:
  - Training loss (must be decreasing after warmup)
  - Eval loss (checkpoint when it stops decreasing)
  - GPU utilization (should be > 85%)
  - Estimated time remaining
- Run quick domain eval every 500 steps:
  - Sample 20 examples from your custom benchmark
  - Generate with greedy decoding; execute against Postgres
  - Log `domain_exec_accuracy` to W&B
- **Terminate instance when training completes**

**Expected training time:** 25K examples × 2 epochs ÷ (32 effective batch) ≈ 1,562 steps. At H100 throughput with LoRA ≈ 250 steps/hour → ~6 hours. Budget: ~$17.

**Acceptance criteria:**
- Training completes without error
- Final eval loss < starting eval loss
- `domain_exec_accuracy` at end of training ≥ Phase 5 baseline + 5 percentage points

**Deliverable:** SFT checkpoint pushed to HuggingFace as `<your-handle>/qwen2.5-coder-7b-postgres-sft-v3`.

---

## Task 4 — Checkpoint Evaluation

**Goal:** Quantify what the SFT step achieved.

**Requirements:**
Run `eval_sft.py` that evaluates:
- Your custom 200-example PostgreSQL/TimescaleDB benchmark (execution accuracy)
- 100-example sample from BIRD-SQL dev set (execution accuracy)
- Phase 5 GRPO model on the same examples (for fair comparison)
- Base Qwen2.5-Coder-7B on the same examples (lower bound)

Report results in `sft_eval_results.md`:

| Model | Custom benchmark | BIRD-SQL sample |
|-------|-----------------|-----------------|
| Base Qwen2.5-Coder-7B | X% | X% |
| Phase 5 GRPO model | X% | X% |
| Week 58 SFT-v3 | X% | X% |

**Deliverable:** `sft_eval_results.md` committed.

---

## Stretch Goals

- Train a second run with rank=64 for 500 steps; compare val loss curves to rank=32. Which converges faster?
- Implement per-skill evaluation: compute execution accuracy separately for each of your 12+ skill categories. Identify which skills improved most vs. least.
- Check for instruction-following regression: run a general instruction-following benchmark (use 20 examples from AlpacaEval or MT-Bench) to verify the SFT run didn't hurt the model's general capability.
