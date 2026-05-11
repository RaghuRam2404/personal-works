# Week 24 Assignment — SOTA Pretraining Recipes Comparison

## Setup Checklist

- [ ] Papers downloaded as PDFs or accessible via arXiv:
  - Llama 3: arxiv.org/abs/2407.21783
  - Qwen2.5: arxiv.org/abs/2412.15115
  - Qwen2.5-Coder: arxiv.org/abs/2409.12186
  - DeepSeek-V3: arxiv.org/abs/2412.19437
  - DeepSeek-Coder: arxiv.org/abs/2401.14196
- [ ] Text editor ready for note-taking
- [ ] No GPU needed this week

---

## Task 1 — Create the Architecture Comparison Table

**Goal:** Extract key architectural facts from each paper into a single reference table.

**Requirements:**

Create `week-24-sota-comparison.md` with a Markdown table containing these rows for each model:

| Dimension | Llama 3-8B | Qwen2.5-7B | Qwen2.5-Coder-7B | DeepSeek-V3 | DeepSeek-Coder-V2 |
|---|---|---|---|---|---|
| Params (total) | | | | | |
| Params (active) | | | | | |
| Training tokens | | | | | |
| Attention type | | | | | |
| FFN type | | | | | |
| Context length | | | | | |
| Vocabulary size | | | | | |
| Tokenizer | | | | | |
| Special features | | | | | |

For each cell, find the exact value from the paper. If not stated explicitly, note "not stated" — do not guess.

**Deliverable:** The completed table in `week-24-sota-comparison.md`.

---

## Task 2 — Data Strategy Comparison

**Goal:** Understand how each model approaches training data.

**Requirements:**

Write 2–3 paragraphs (not a table — prose) comparing the data strategies of the 5 models. Your comparison must address:
- Total token count and approximate source breakdown (web, code, math, books, etc.)
- What quality filtering was applied (did they use LLM judges for quality? What heuristics?)
- How they handle code vs. natural language mixing
- Which model's data strategy you would most want to emulate if building a domain code model

**Deliverable:** Add a "Data Strategy Comparison" section to `week-24-sota-comparison.md`.

---

## Task 3 — Post-Training Pipeline Analysis

**Goal:** Understand the post-training stages each model uses.

**Requirements:**

For each model, extract and summarize the post-training pipeline. Use this template for each model:

```
Model: [name]
Stage 1: [SFT / continued pretraining / domain pretraining]
  - Dataset size: 
  - Objective:
Stage 2: [RLHF / DPO / GRPO / none]
  - Algorithm:
  - Reward source:
Stage 3 (if any): [specialized fine-tuning]
```

Write 1–2 sentences commentary per model on what is notable about their post-training approach.

**Deliverable:** Add a "Post-Training Pipelines" section to `week-24-sota-comparison.md`.

---

## Task 4 — Model Selection Recommendation

**Goal:** Produce a reasoned recommendation for your Phase 6 fine-tuning base model.

**Requirements:**

Write a 400-word section titled "My Phase 6 Base Model Choice" that answers:
1. Which of the 5 models (or their variants) would you choose to fine-tune for PostgreSQL/TimescaleDB text-to-SQL?
2. Why is that model better than the alternatives for your specific task?
3. What are the 2–3 biggest risks or limitations of your chosen model for this task?
4. What pre-existing SQL knowledge does your chosen model have that reduces the fine-tuning burden?

**Expected answer:** Qwen2.5-Coder-7B (or DeepSeek-Coder-V2-Lite). Both are defensible. If you choose a different model, your justification must be convincing.

**Deliverable:** Full `week-24-sota-comparison.md` document (minimum 800 words).

GitHub commit: `week-24-sota-recipes`

---

## Stretch Goals

- Find the GitHub repositories for each model's training code (if open-sourced) and compare what is actually public vs. what is described in the paper
- Investigate which models are licensed for commercial use (Apache 2.0, Llama community license, Qwen license) — this matters for a real product
- Read the MoE section of DeepSeek-V3 in detail and write a 200-word explanation of why MoE reduces active compute compared to a dense model
