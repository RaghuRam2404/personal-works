# Week 72 Assignment — Frontier Reading: DeepSeek and Qwen

## Setup Checklist

- [ ] Download PDFs: DeepSeek-V3 (arXiv 2412.19437), DeepSeek-R1 (arXiv 2501.12948), Qwen2.5 technical report (arXiv 2412.15115)
- [ ] `reading_notes/` directory from Week 71 available
- [ ] Your Week 75 candidates list ready: Llama 3.1 8B, Gemma 2 9B, DeepSeek-Coder-V2-Lite, DeepSeek-R1-Distill-Qwen-7B

## Task 1: DeepSeek-V3 — Architecture Extraction

**Goal:** Understand MoE and MLA well enough to evaluate DeepSeek-Coder-V2-Lite as a Week 75 candidate.

**Requirements:**
- [ ] Explain Multi-head Latent Attention (MLA) in your own words: what is compressed, by how much, and what is the accuracy tradeoff?
- [ ] Explain the MoE routing mechanism: how many experts total, how many active per token, what is the routing function?
- [ ] Find the KV cache memory comparison between MLA and standard GQA at the same context length: how many GB difference at 4096 context?
- [ ] Assess: would DeepSeek-Coder-V2-Lite's MoE architecture cause problems with Unsloth/PEFT LoRA fine-tuning? Look up Unsloth's model support page.
- [ ] Save as `reading_notes/deepseekv3_notes.md`

**Deliverable:** `reading_notes/deepseekv3_notes.md` (300–400 words)

## Task 2: DeepSeek-R1 — GRPO and Distillation

**Goal:** Understand how R1's GRPO differs from yours and evaluate R1-Distill-Qwen-7B for Week 75.

**Requirements:**
- [ ] Describe R1's three training stages (cold start, GRPO RL, rejection sampling + SFT distillation) in 2 sentences each
- [ ] Compare R1's GRPO hyperparameters to yours (Week 60): K value, steps, reward types
- [ ] Find the accuracy of DeepSeek-R1-Distill-Qwen-7B on the benchmarks R1 reports: how does it compare to Qwen2.5-Coder-7B-Instruct (your current base) on any shared benchmark?
- [ ] Assess: is R1-Distill-Qwen-7B a better starting point than Qwen2.5-Coder-7B-Instruct for SQL fine-tuning? Argue for or against in 3 sentences.
- [ ] Save as `reading_notes/deepseekr1_notes.md`

**Deliverable:** `reading_notes/deepseekr1_notes.md` (300–400 words)

## Task 3: Qwen2.5 — Understanding Your Base Model

**Goal:** Understand what Qwen2.5-Coder-7B already knows so you can stop re-teaching it things in Week 75.

**Requirements:**
- [ ] Find the data mixture breakdown for Qwen2.5: what fraction is code? SQL specifically?
- [ ] Find the Spider 1.0 or BIRD-SQL score for Qwen2.5-Coder-7B-Instruct without any fine-tuning (this is your Week 62 base model baseline — does the paper's number match yours?)
- [ ] Identify what the Coder post-training adds on top of the base model: what tasks were in the Coder SFT data?
- [ ] Identify one thing the Qwen2.5 post-training did that may conflict with your domain-specific DPO: does general RLHF make domain-specific DPO harder or easier?
- [ ] Save as `reading_notes/qwen25_notes.md`

**Deliverable:** `reading_notes/qwen25_notes.md` (300–400 words)

## Task 4: Week 75 Base Model Decision Matrix

**Goal:** A structured comparison of all four Week 75 candidate base models.

**Requirements:**
- [ ] Create `reading_notes/week75_base_model_decision.md` with a table:
  | Model | Active Params | Spider EM | SQL Tokenization | Unsloth Support | Quant Complexity | Verdict |
  |-------|--------------|-----------|-----------------|-----------------|------------------|---------|
- [ ] Fill in each row for: Llama 3.1 8B, Gemma 2 9B, DeepSeek-Coder-V2-Lite, DeepSeek-R1-Distill-Qwen-7B
- [ ] Write a 100-word recommendation: which model you will use in Week 75 and why
- [ ] Your recommendation must consider: toolchain compatibility, expected SQL capability, quantization complexity, and inference efficiency

**Deliverable:** `reading_notes/week75_base_model_decision.md`

## Stretch Goals

- Run Qwen2.5-Coder-7B and DeepSeek-R1-Distill-Qwen-7B side-by-side on 10 TimescaleDB SQL questions without fine-tuning; compare zero-shot accuracy as a preview for Week 75
- Read the DeepSeek-R1 distillation section carefully: reproduce their "rejection sampling" pipeline on 50 of your SQL examples to see what the R1-style reasoning chain looks like for SQL problems
