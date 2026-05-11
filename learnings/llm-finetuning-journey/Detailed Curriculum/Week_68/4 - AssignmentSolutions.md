# Week 68 Assignment Solutions

## Task 1: Extracting Hyperparameters

W&B stores every config key in the run's Config tab. The tricky part is that some keys differ between libraries (Unsloth uses different naming than plain HuggingFace Trainer). Map them:

```
W&B key                → Paper name
per_device_train_bs    → batch size per GPU
gradient_accum_steps   → gradient accumulation
learning_rate          → peak LR
lr_scheduler_type      → LR schedule
warmup_steps           → warmup steps
num_train_epochs + steps_per_epoch → training steps
max_seq_length         → max sequence length
lora_r                 → LoRA rank
lora_alpha             → LoRA alpha
lora_target_modules    → LoRA target modules
```

Effective batch size = `per_device_train_bs × gradient_accum_steps × num_GPUs`.

## Task 2: Loss Formula Snippets

Include these exact formulas in your training section (they signal to reviewers that you understand what you trained):

```
# CPT loss (standard causal LM)
L_CPT = -E[log p_θ(x_t | x_{<t})]

# SFT loss (same, but only on assistant turns)
L_SFT = -E[log p_θ(y_t | x, y_{<t})]
# where x is the prompt, y is the SQL response

# DPO loss
L_DPO = -E[log σ(β log(p_θ(y_w|x)/p_ref(y_w|x))
              - β log(p_θ(y_l|x)/p_ref(y_l|x)))]
# β=0.1, y_w = chosen SQL, y_l = rejected SQL

# GRPO reward (your executable-SQL reward from Week 60)
r(y) = 1.0   if execute(y) == execute(y_gold)
       0.5   if y is valid SQL but wrong result
       0.0   if y is not executable
```

## Task 3: Architecture Decisions Template

```markdown
## LoRA Configuration

We apply LoRA (Hu et al. 2022) to all linear projection layers in
attention (Q, K, V, O projections) and the feed-forward network
(gate, up, and down projections), totaling 32 × 7 = 224 adapter
matrices per layer across 32 transformer layers.

LoRA rank r=64 was selected based on a validation accuracy sweep
over r ∈ {16, 32, 64, 128}:

| LoRA rank | SFT val loss | Custom 200 accuracy |
|-----------|-------------|---------------------|
| 16        | 0.312       | 76.5%               |
| 32        | 0.287       | 79.1%               |
| 64        | 0.271       | 83.1%               |
| 128       | 0.269       | 83.0%               |

Rank 64 provides the best accuracy-to-parameter-count tradeoff.
We set alpha=128 (2× rank) following QLoRA (Dettmers et al. 2023).
```

## Task 4: Assembling the Draft

```markdown
# postgres-sqlcoder-7b Technical Report

## Abstract
[from report/abstract_draft.md]

## 1. Introduction
[from report/introduction.md]

## 2. Dataset Construction
[from report/dataset_section.md]

## 3. Training Pipeline
[from report/training_section.md + architecture_decisions.md]
```

Apply a final consistency pass with this checklist:
- Every "25,500" in Section 3 matches "25,500" in Section 4 SFT reference
- Every "102M tokens" in Section 3 matches "102M tokens" in Section 4 CPT
- All table numbers (Table 1, Table 2...) are sequential and referenced in text

## Common Gotchas

- DPO beta is often logged as `beta` in W&B but is called "β" in the paper; confirm the value is the actual beta, not a learning rate.
- GRPO K (number of samples per prompt) may be logged as `num_generations`; confirm and use the right number.
- "GPU-hours" = number of GPUs × hours of wall-clock time; make this calculation explicit in your compute section.
- If you ran multiple trial runs before the final run, report only the final run's hyperparameters; mention failed runs briefly in a footnote.

## How to Verify You Did It Right

Take one hyperparameter (e.g., SFT learning rate). Trace it from: W&B config tab → `report/hyperparams_table.md` → `report/training_section.md` body text → the hyperparameter table in the section. All four should show the same value. If any differ, you have a consistency error.
