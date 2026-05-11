# Week 51 Assignment — Iteration Run 2 and Final Model Selection

## Setup Checklist

- [ ] Week 50 iteration results (v3-iter1 model, iteration log)
- [ ] Second failure mode hypothesis from Week 50 diagnosis.md
- [ ] All previous model checkpoints accessible (v1, v2, v3, v3-iter1)
- [ ] 200-query held-out eval set (the same one used throughout Phase 5)
- [ ] Remaining RunPod budget: ~$15–25

---

## Task 1 — Run the Second Targeted Experiment

**Goal:** Address the second-priority failure mode from your Week 50 diagnosis.

**Requirements:**
- Start from your best checkpoint (likely v3-iter1 from Week 50)
- Make the specific change identified for the second failure mode
- Run 200–400 GRPO steps
- Monitor: the same W&B metrics as before
- Stopping criterion: if reward_std drops below 0.05 for 50 steps, stop early

**Deliverable:**
- `<your-handle>/postgres-sqlcoder-7b-v3-iter2` on HF Hub
- `week-51-iteration/experiment2_log.md` (hypothesis → result format from Week 50)

---

## Task 2 — Final Evaluation Sweep

**Goal:** Run evaluation on ALL models from Phase 5 using the same held-out test set.

**Requirements:**
- Models to evaluate: v1, v2, v3, v3-iter1, v3-iter2
- Use the exact same 200-query held-out test set for all models
- Use greedy decoding (temperature=0, do_sample=False) for all models
- Report for each model:
  - Overall execution accuracy
  - Overall semantic accuracy
  - Syntax error rate
  - Complex query execution accuracy (top 50 hardest queries)
  - Mean generation length (tokens)
- Produce a single Markdown table with all 5 metrics × 5 models = 25 cells

**Deliverable:** `week-51-iteration/final_eval_all_models.md`

---

## Task 3 — Best Model Selection

**Goal:** Apply the model selection framework to pick the best checkpoint.

**Requirements:**
- Document your decision using the 5-criterion framework from the curriculum:
  1. Execution accuracy (primary)
  2. Semantic accuracy
  3. Complex query performance
  4. KL divergence from SFT (use W&B final step KL for each model)
  5. Mean generation length
- Write 3–5 sentences explaining the tradeoffs between the top 2 candidates
- Push the selected model to HF Hub with tag: `<your-handle>/postgres-sqlcoder-7b-phase5-best`

**Deliverable:** 
- `week-51-iteration/model_selection.md`
- `<your-handle>/postgres-sqlcoder-7b-phase5-best` on HF Hub

---

## Task 4 — Phase 5 Final Eval Report

**Goal:** Write the complete Phase 5 progress story for the Gate submission.

**Requirements:**
- File: `week-51-iteration/phase5_final_report.md`
- Sections:
  1. **Executive Summary** (3 sentences: what you did, what improved, what remains)
  2. **Model Progression Table** (all 5 models × all 5 metrics)
  3. **Key Findings** (3–5 bullet points: what worked, what didn't, surprises)
  4. **Residual Gaps** (what v3 still cannot do well)
  5. **Phase 6 Roadmap** (what the larger dataset and longer training in Phase 6 should address)
- Length: 400–600 words

---

## Stretch Goals

- Generate 10 responses from your best model on Phase 6-style prompts (TimescaleDB hyperfunctions, multi-step analytics, correlated subqueries). Manually evaluate. Where does it still fail?
- Create a model card on HuggingFace Hub for `postgres-sqlcoder-7b-phase5-best` documenting: training procedure, eval results, known limitations, and intended use.
- Compare your best model's output on 3 queries to GPT-4o. For which query types is your model already competitive? (This previews the Phase 6 goal of beating GPT-4 on your domain.)
