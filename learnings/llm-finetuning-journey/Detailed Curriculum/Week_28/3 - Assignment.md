# Week 28 Assignment — Post-Training Pipeline Diagram

## Setup Checklist

- [ ] GitHub repo open and accessible (your Phase 3 repo is fine)
- [ ] Excalidraw account (free at excalidraw.com) OR pen/paper + camera
- [ ] InstructGPT paper downloaded: https://arxiv.org/abs/2203.02155
- [ ] Karpathy video queued: https://www.youtube.com/watch?v=7xTGNNLPyMI
- [ ] No GPU needed this week

---

## Task 1 — Watch and Annotate Karpathy's Video

**Goal:** Extract the core post-training concepts from Karpathy's walkthrough and connect them to your prior Phase 3 knowledge.

**Requirements:**
- Watch the full 3h31m video: [Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI)
- While watching, keep a notes file called `week28_karpathy_notes.md` in your repo
- Write down at least 15 distinct facts or insights (one per note entry)
- For at least 5 notes, write a follow-up sentence connecting the insight to something you built in Phase 2 or 3 (e.g., "Karpathy says base models are document completers — this matches my nanoGPT which just autocompletes tokens")

**Deliverable:** `week28_karpathy_notes.md` committed to your GitHub repo.

**Hints:**
- Pause at the 30-minute mark and write a one-paragraph summary of what you've learned before continuing
- When Karpathy discusses RLHF, make sure you understand why the SFT step comes before preference optimization

---

## Task 2 — Read InstructGPT Sections 1–3

**Goal:** Understand the original SFT + RLHF pipeline that produced the first generation of instruction-following LLMs.

**Requirements:**
- Read sections 1, 2, and 3 of [InstructGPT: Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155)
- Write a summary (200–400 words) in `week28_instructgpt_summary.md` that answers:
  - What dataset did OpenAI use for the SFT stage? How many examples?
  - What is the reward model trained on, and what is its input/output?
  - Why did the authors need both SFT and RLHF? Why wasn't SFT alone sufficient?
  - How does their pipeline map to the SFT → DPO → GRPO pipeline you will build?

**Deliverable:** `week28_instructgpt_summary.md` committed.

---

## Task 3 — Draw the Post-Training Pipeline

**Goal:** Create a reference diagram you will reuse throughout Phase 4 and 5.

**Requirements:**
- Draw the full post-training pipeline with at minimum these nodes:
  - Pretrained base model
  - SFT stage (inputs, outputs, loss type)
  - DPO/preference stage (inputs, outputs, what it optimizes)
  - GRPO/RLVR stage (reward signal source for SQL correctness)
  - Deployed model
- For each stage, annotate: (a) typical dataset size, (b) compute cost relative to SFT, (c) what the model gains
- Include a branch showing the alternative paths: "continued pretraining" before SFT if domain vocab is missing
- Use Excalidraw (export as PNG) or hand-draw and photograph

**Deliverable:** `week28_pipeline_diagram.png` committed to GitHub. Commit message: `week-28-pipeline-diagram`.

---

## Task 4 — Three-Column Comparison Table

**Goal:** Solidify the distinction between the three approaches.

**Requirements:**
- Create a markdown table in `week28_comparison.md` with rows for:
  - Training objective (loss function)
  - Typical data format
  - Typical data size
  - Compute cost
  - What the model learns
  - When to use
  - Risk of catastrophic forgetting (low/medium/high)
- Columns: Continued Pretraining | SFT | Instruction Tuning
- Fill every cell with a concrete answer, not "varies"

**Deliverable:** `week28_comparison.md` committed.

---

## Stretch Goals

- Watch Karpathy's older 2-hour "Let's build ChatGPT from scratch" video and compare his 2023 vs. 2024/2025 framing of the post-training pipeline. What changed?
- Read the abstract and introduction of the [Llama 3 technical report](https://arxiv.org/abs/2407.21783) and identify which stages of the post-training pipeline Meta used.
- Read Karpathy's [2025 LLM Year in Review blog post](https://karpathy.bearblog.dev/year-in-review-2025/) and write 3 bullet points on what surprised you most.
