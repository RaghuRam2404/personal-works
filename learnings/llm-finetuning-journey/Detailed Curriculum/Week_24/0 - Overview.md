# Week 24 Overview — Reading Week: SOTA Pretraining Recipes (2024–2025)

This file is your map for Week 24. Read it first; everything else fits inside it.

## The story this week

Reading papers is a skill. Most engineers read only abstracts and conclusions; senior engineers read selectively but deeply. This week, you practice reading 5 major technical reports at different depths.

## What you need to do

- [ ] Papers downloaded as PDFs or accessible via arXiv:
  - Llama 3: arxiv.org/abs/2407.21783
  - Qwen2.5: arxiv.org/abs/2412.15115
  - Qwen2.5-Coder: arxiv.org/abs/2409.12186
  - DeepSeek-V3: arxiv.org/abs/2412.19437
  - DeepSeek-Coder: arxiv.org/abs/2401.14196
- [ ] Text editor ready for note-taking
- [ ] No GPU needed this week

Concretely, by the end of the week you should be able to:

- Identify the key architectural and training differences between Llama 3, Qwen2.5, Qwen2.5-Coder, DeepSeek-V3, and DeepSeek-Coder
- Explain what "post-training" pipeline means and how it differs across these models
- Compare their data strategies (sources, mixing ratios, filtering approaches)
- Summarize the compute scales and efficiency innovations used in each recipe
- Form a reasoned opinion on which base model is the best starting point for your PostgreSQL fine-tuning goal

## Suggested order through the files

1. **1 - Curriculum.md** — read first. Concepts, connections, common pitfalls.
2. **2 - Resources.md** — papers, videos, blog posts, repos, docs to consult while studying.
3. **3 - Assignment.md** — the hands-on task you must complete this week.
4. **4 - AssignmentSolutions.md** — only after you have attempted the assignment yourself.
5. **5 - Quiz.md** — self-test that you have absorbed the week's material.
6. **6 - Answers.md** — quiz answers and explanations; check after attempting.
7. **7 - Glossary.md** — terminology used this week, indexed for fast lookup.
8. **8 - TakeAway.md** — the one-page summary you keep for the rest of the course.

## Time budget

- 1h: Skim Llama 3 paper (focus on data, architecture table, benchmark table)
- 1.5h: Deep-read Qwen2.5-Coder paper (all sections)
- 1.5h: Deep-read DeepSeek-Coder paper (all sections)
- 1h: Skim Qwen2.5 and DeepSeek-V3 papers
- 2h: Write the 3-page comparison document (`week-24-sota-comparison.md`)

## Why this week matters

**One-liner:** Qwen2.5-Coder-7B (5.5T code tokens) is your fine-tuning base; MoE enables large total params with small active compute; GQA shrinks KV cache.
