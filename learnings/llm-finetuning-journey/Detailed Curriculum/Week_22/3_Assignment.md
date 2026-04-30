# Week 22 Assignment — Evaluate the 50M Language Model

## Setup Checklist

- [ ] Best checkpoint from Week 21 available locally or downloaded from HuggingFace Hub
- [ ] `val.bin` available (from Week 20 data pipeline)
- [ ] Model can be loaded with `torch.load` and runs a forward pass
- [ ] W&B run from Week 21 is accessible for reference

---

## Task 1 — Compute Validation Perplexity

**Goal:** Compute the exact per-token cross-entropy and perplexity on your held-out validation set.

**Requirements:**

Implement `eval.py` with a `compute_perplexity(model, val_path, block_size=1024, n_batches=200)` function.

- Load model from checkpoint
- Load `val.bin` as a memory-mapped array
- Run non-overlapping windows of size `block_size=1024`
- Use at least 200 windows (200K tokens of evaluation)
- Report: mean cross-entropy loss, perplexity, and 95% confidence interval (std of per-batch losses)

Additionally:
- Compute perplexity on a short (1,000-token) SQL snippet of your choice (any PostgreSQL documentation text)
- Compare: does your model assign lower perplexity to general English or to SQL?

**Deliverable:** `pretrain-50m/eval.py`. A printout showing:
```
Val perplexity (FineWeb-Edu val, 200 batches): 31.4 ± 2.1
SQL snippet perplexity: 58.2
→ Model knows 1.85× less about SQL than general English (expected at this stage)
```

**Acceptance criteria:** Val perplexity is between 15 and 60. If outside this range, diagnose before proceeding.

---

## Task 2 — Generate Text Samples

**Goal:** Qualitatively evaluate your model's language generation quality.

**Requirements:**

Implement `generate.py` with a sampling function that supports temperature, top_k, and top_p.

Generate 12 samples using these prompts and settings:

| # | Prompt | Temperature | top_k |
|---|---|---|---|
| 1 | "The history of" | 0.8 | 50 |
| 2 | "The history of" | 1.2 | 50 |
| 3 | "The history of" | 0.3 | 50 |
| 4 | "Scientists recently discovered" | 0.8 | 50 |
| 5 | "SELECT * FROM" | 0.8 | 50 |
| 6 | "CREATE TABLE users" | 0.8 | 50 |
| 7 | "PostgreSQL is a" | 0.8 | 50 |
| 8 | "The best way to optimize a SQL query is" | 0.8 | 50 |
| 9 | "Once upon a time in a land" | 0.8 | 50 |
| 10 | "In 2023, artificial intelligence" | 0.8 | 50 |
| 11 | "The capital of France" | 0.5 | 20 |
| 12 | "The capital of France" | 0.8 | 100 |

For each sample:
- Generate 150 tokens
- Include the full generated text in your report
- Rate it on fluency (1–5) and coherence (1–5)

**Deliverable:** `pretrain-50m/generate.py` + a section in `week-22-evaluation.md` with all 12 samples and ratings.

---

## Task 3 — Failure Analysis

**Goal:** Identify specific weaknesses in your model's output.

**Requirements:**

Write a 400-word failure analysis in `week-22-evaluation.md` that identifies at least 4 concrete failure modes. For each:
- Describe the failure with an example from your generated samples
- Explain the likely mechanism (why does a 50M model do this?)
- State what would fix it (more data? More params? Better data quality? Fine-tuning?)

Expected failure modes to find:
- Repetition loops
- Loss of topic coherence after ~30 tokens
- SQL generation that produces syntactically invalid queries
- Hallucinated entities or dates
- Inability to complete common facts ("The capital of France is Paris")

---

## Task 4 — Evaluation Report

**Goal:** Write a complete, honest model evaluation that a colleague could read and understand.

**Requirements:**

Write `week-22-evaluation.md` (minimum 800 words) with these sections:

1. **Training Summary** (links, numbers, no fluff)
2. **Perplexity Analysis** (your numbers + comparison to GPT-2-small on similar data)
3. **Sample Gallery** (the 12 samples with ratings from Task 2)
4. **Failure Analysis** (from Task 3)
5. **Scaling Takeaways** (what would most improve this model? Support with evidence from your training run)
6. **Fine-Tuning Readiness** (is this model a good base for PostgreSQL fine-tuning? Compare to the alternative: fine-tuning Qwen2.5-Coder-7B directly)

For section 6, the expected answer is that your 50M model is NOT a good base for fine-tuning — you should fine-tune Qwen2.5-Coder-7B instead. State this clearly and justify it.

**Deliverable:** `pretrain-50m/week-22-evaluation.md`

GitHub commit: `week-22-evaluation`

---

## Stretch Goals

- Plot the loss curve over the 200 validation batches as a histogram. Does the loss distribution look Gaussian or is it skewed?
- Implement beam search (beam width=5) and compare to sampling — which produces better SQL?
- Run your model on BPT (the BabyLM Evaluation Suite) and compare to their baselines
