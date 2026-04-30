# Week 45 Assignment — Apply DPO to postgres-sqlcoder-7b-v1

## Setup Checklist

- [ ] Colab Pro with A100 (40GB) GPU
- [ ] Packages: `unsloth`, `trl>=0.9.0`, `transformers`, `datasets`, `peft`, `wandb`
- [ ] Your model: `postgres-sqlcoder-7b-v1` (or the HF Hub path from Phase 4)
- [ ] Your dataset: `<your-handle>/postgres-sql-preferences-v1` (from Week 44)
- [ ] Held-out test set: 200 prompts with reference SQL (NOT used in preference labeling)
- [ ] Postgres DB accessible from Colab (use `ngrok` or a cloud Postgres instance)

---

## Task 1 — Run DPO Training

**Goal:** Fine-tune postgres-sqlcoder-7b-v1 with DPO on your SQL preference data.

**Requirements:**
- Base: `postgres-sqlcoder-7b-v1` (your Phase 4 checkpoint)
- Dataset: `<your-handle>/postgres-sql-preferences-v1`
- Training config:
  - β = 0.1 (starting point)
  - Learning rate: 5e-7
  - Batch size: 2 per device, gradient accumulation: 8
  - LoRA: r=16, α=32, target_modules="all-linear"
  - Max prompt length: 512, max completion length: 256
  - Epochs: 1 (use 2 only if reward_margin < 0.3 after epoch 1)
  - W&B project: `week-45-dpo-sql`
- Log every 25 steps: `reward_margin`, `rewards/chosen`, `rewards/rejected`, `loss`
- Save checkpoint at end of training

**Deliverable:**
- Trained model pushed to HF Hub: `<your-handle>/postgres-sqlcoder-7b-v2-dpo`
- W&B run link in `week-45-dpo/training_notes.md`
- GitHub commit: `week-45-dpo-sql`

---

## Task 2 — Evaluation: v1 vs. v2

**Goal:** Produce a quantitative eval report comparing SFT-only model vs. DPO model.

**Requirements:**
- Use your 200-query held-out test set (not used in preference labeling)
- For each model (v1 and v2), generate SQL for all 200 prompts
  - Use greedy decoding (temperature=0, do_sample=False)
  - Max new tokens: 256
- Run each generated SQL through your `execute_sql()` harness
- Report (for each model):
  - Execution accuracy: % of queries that execute without error
  - Semantic accuracy: % of queries that return the same rows as reference SQL
  - Syntax error rate
  - Empty result rate (executes but returns 0 rows)
- Break down results by query complexity:
  - Simple (single table, no JOIN): ≥ 50 examples
  - Medium (1–2 JOINs): ≥ 50 examples
  - Complex (subqueries, CTEs, 3+ JOINs): ≥ 50 examples

**Acceptance criterion:** v2 execution accuracy > v1 execution accuracy on the full test set.

**Deliverable:** `week-45-dpo/eval_report.md` — a Markdown table with all metrics

---

## Task 3 — Training Diagnostics

**Goal:** Document what went wrong (if anything) and why.

**Requirements:**
- Write `week-45-dpo/training_notes.md` with:
  - Final reward_margin (and whether it was > 0.5)
  - Whether any of the 4 failure modes from the curriculum occurred
  - If v2 does NOT beat v1: your diagnosis and planned fix
  - If v2 DOES beat v1: by how many percentage points? On which query types?

---

## Stretch Goals

- Try β = 0.05 and β = 0.2 for one epoch each. Compare reward_margin and eval accuracy. Which β gives the best semantic accuracy on complex queries?
- Apply [length normalization](https://huggingface.co/docs/trl/dpo_trainer#length-normalized-dpo) in TRL and compare with unnormalized DPO. Does it help on long complex queries?
- Generate 10 examples from v1 and v2 for the same hard prompt (3-table JOIN with HAVING). Read them manually. Does v2 produce visibly better SQL?
