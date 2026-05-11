# Week 68 Assignment — Technical Report: Training Pipeline Section

## Setup Checklist

- [ ] W&B project open for all four training runs (Weeks 57–60)
- [ ] `report/introduction.md` and `report/dataset_section.md` from Week 67 ready
- [ ] Exact hyperparameter configs available (W&B config tab, or your training scripts)
- [ ] GPU-hour logs or RunPod billing dashboard for compute budget

## Task 1: Extract Hyperparameters from W&B

**Goal:** A single source-of-truth hyperparameter table that all four sections will reference.

**Requirements:**
- [ ] Open each W&B run (CPT, SFT, DPO, GRPO) and go to the Config tab
- [ ] Record: LR, LR schedule, warmup steps, batch size, gradient accumulation steps, LoRA rank, LoRA alpha, LoRA target modules, max seq length, training steps, optimizer, weight decay, gradient clipping
- [ ] Build `report/hyperparams_table.md` as a markdown table with all four stages as columns
- [ ] Note any hyperparameter that changed mid-run (e.g., LR restart) as a footnote
- [ ] Cross-check: effective batch size = batch_size × gradient_accumulation_steps — report the effective value

**Deliverable:** `report/hyperparams_table.md`

## Task 2: Write the Training Pipeline Section

**Goal:** A 600–900 word training pipeline section with all four stages, hyperparameter table, and compute budget.

**Requirements:**
- [ ] Section 4.1: Base model paragraph — model name, parameters, why chosen, license
- [ ] Section 4.2: CPT — objective (write the loss formula), dataset, hyperparameters, duration, measured perplexity drop
- [ ] Section 4.3: SFT — objective, dataset reference to Section 3, hyperparameters, final validation loss
- [ ] Section 4.4: DPO — objective (write the DPO loss formula), beta value, dataset, training duration, reward margin achieved
- [ ] Section 4.5: GRPO — objective (write the reward function in pseudocode or math), K value, training steps, accuracy gain over DPO baseline
- [ ] Section 4.6: Compute budget — GPU-hours per stage, total GPU-hours, approximate cost, minimum hardware to reproduce
- [ ] Include the hyperparameter table from Task 1 in the section body
- [ ] All numbers must match W&B logs exactly

**Deliverable:** `report/training_section.md`

## Task 3: Write Architecture Decisions Subsection

**Goal:** A 200–300 word justification for each non-obvious architectural choice.

**Requirements:**
- [ ] LoRA rank selection: why r=64 specifically (reference any ablation you ran)
- [ ] Target modules: why all-linear vs attention-only (latency vs accuracy tradeoff)
- [ ] Sequence length: why 2048 for CPT/SFT and why 1024 for DPO/GRPO
- [ ] Chat template: which template, why (match your training format)
- [ ] If you have ablation data for any of these choices, cite the result (e.g., "r=32 gave 1.8 pp lower accuracy on SFT val set")

**Deliverable:** `report/architecture_decisions.md` (this will be integrated into Section 4 or Appendix)

## Task 4: Integrate into Main Report

**Goal:** A single `report/report_draft_v1.md` that assembles Abstract through Section 4.

**Requirements:**
- [ ] Assemble: Abstract + Introduction + Dataset + Training into one document
- [ ] Check cross-references: every table number, every section reference
- [ ] Check number consistency: every number in Section 4 matches Section 3 (e.g., SFT dataset size matches)
- [ ] Estimated word count: target 2,000–2,800 words through Section 4
- [ ] Proofread for past-tense consistency (methods: past tense; paper overview: present tense)

**Deliverable:** `report/report_draft_v1.md`

## Stretch Goals

- Write the complete Appendix hyperparameter section with every single training detail, formatted as the NeurIPS reproducibility appendix expects
- Create a training curves figure: plot val loss vs steps for SFT, and reward vs steps for GRPO, save as `report/figures/training_curves.png`
- Write a "training instability" subsection documenting any runs that failed or required restarts, and what you learned from them
