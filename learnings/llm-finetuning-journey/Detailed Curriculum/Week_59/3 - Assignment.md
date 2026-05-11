# Week 59 Assignment — DPO on Refreshed Preference Dataset

## Setup Checklist

- [ ] SFT-v3 checkpoint accessible: `<your-handle>/qwen2.5-coder-7b-postgres-sft-v3`
- [ ] PostgreSQL with test schemas running
- [ ] Colab Pro with A100 (or RunPod A100 if needed — ~$4 for 2 hours)
- [ ] `trl` library with DPOTrainer: `pip install trl>=0.8`
- [ ] W&B project `week-59-dpo` created

---

## Task 1 — Build the Refreshed Preference Dataset

**Goal:** Generate 5K "hard" preference pairs using your SFT-v3 model.

**Requirements:**
Write `build_dpo_dataset.py` that:
- Samples 1,000 prompts from v3 dataset (stratified: 40% TimescaleDB, 60% standard PostgreSQL)
- For each prompt, generates 8 SQL candidates using the SFT-v3 model with temperature=0.8
- Executes all 8 candidates against Postgres; records execution_status and result_hash
- Labels pairs:
  - chosen = the candidate that executes correctly AND matches the reference result hash
  - rejected = the candidate that: (a) executes incorrectly (wrong rows), preferred over (b) syntax error, which is preferred over (c) didn't parse
  - If ALL candidates execute correctly: chosen = best (lowest AST depth or exact match to reference), rejected = the one with the most redundant SQL
  - If NO candidate executes correctly: skip this prompt
- Saves as JSONL with fields: `prompt`, `chosen`, `rejected`, `difficulty`, `skill`, `chosen_executes`, `rejected_executes`
- Target: at least 4,000 valid pairs (expect ~40% of prompts to yield usable pairs)

**Audit step (after generation):**
Manually inspect 50 random pairs. Count how many are "hard" (both chosen and rejected execute, but chosen is meaningfully better). Target: > 50% of your 50-sample should be hard pairs. If < 30%, your pair selection logic needs revision.

**Deliverable:** `dpo_pairs_v3.jsonl` with ≥ 4,000 pairs committed + audit summary in `dpo_audit.md`.

---

## Task 2 — DPO Training Script

**Goal:** Train DPO on your preference dataset.

**Requirements:**
Write `train_dpo.py` using TRL's DPOTrainer:

```python
from trl import DPOTrainer, DPOConfig

dpo_config = DPOConfig(
    beta=0.2,                  # lower than default (0.3) — SQL labels are clean
    learning_rate=5e-5,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,  # effective = 16
    bf16=True,
    logging_steps=10,
    eval_steps=100,
    save_steps=200,
    output_dir="./dpo_output",
    run_name="week-59-dpo",
    report_to="wandb",
)

trainer = DPOTrainer(
    model=model,               # SFT-v3 model with new LoRA
    ref_model=ref_model,       # frozen SFT-v3 (no LoRA, or merged)
    args=dpo_config,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
)
```

Log to W&B:
- DPO loss (should approach 0, not go deeply negative)
- Reward margin: `chosen_reward - rejected_reward` (should be positive and increasing)
- `rewards/chosen` and `rewards/rejected` separately
- Domain execution accuracy every 200 steps

**Deliverable:** `train_dpo.py` committed; DPO run completes on Colab Pro (~2 hours at A100).

---

## Task 3 — DPO Evaluation

**Goal:** Verify DPO improved over the SFT baseline.

**Requirements:**
Run `eval_dpo.py` comparing on your 200-example custom benchmark:

| Model | Exec accuracy | Idiomatic SQL % | TimescaleDB accuracy |
|-------|--------------|-----------------|---------------------|
| SFT-v3 | X% | X% | X% |
| DPO-v3 | X% | X% | X% |

"Idiomatic SQL %" = fraction of correct queries where a SQL expert would call the query "good style" (run 50 examples through your LLM judge at temperature=0.0 and rate for style).

**Acceptance criteria:** DPO-v3 achieves ≥ 2pp improvement on custom benchmark over SFT-v3.

**Deliverable:** `dpo_eval_results.md` committed + DPO checkpoint pushed as `<your-handle>/qwen2.5-coder-7b-postgres-dpo-v3`.

---

## Stretch Goals

- Experiment with beta=0.1 vs beta=0.3: run 500-step pilots at each and compare reward margin and eval accuracy
- Add an "idiomatic SQL" signal to pair selection: use your LLM judge to score the chosen and rejected on style (even if both execute correctly), then DPO pushes toward idiomatic style
- Analyze the DPO model's behavior change: for 10 prompts, compare SFT-v3 and DPO-v3 outputs side-by-side. Does DPO produce noticeably more idiomatic SQL?
