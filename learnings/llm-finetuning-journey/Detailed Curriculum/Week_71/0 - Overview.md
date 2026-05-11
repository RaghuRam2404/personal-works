# Week 71 Overview — Frontier Reading 1: Tulu 3, SmolLM2, OLMo 2

This file is your map for Week 71. Read it first; everything else fits inside it.

## The story this week

Reading an LLM technical report is a skill. You are not reading for full comprehension of every section — you are reading to extract: (a) what they did that is new, (b) what numbers demonstrate the improvement, (c) what is reusable for your own work. A structured reading approach:

## What you need to do

- [ ] Download PDFs: Tulu 3 (arXiv 2411.15124), SmolLM2 (arXiv 2502.02737), OLMo 2 (arXiv 2501.00656)
- [ ] `reading_notes/` directory created in your project repo
- [ ] Your own postgres-sqlcoder-7b technical report nearby for cross-referencing

Concretely, by the end of the week you should be able to:

- Summarize the key training innovations in Tulu 3 (RLVR, on-policy data generation) and relate them to your own GRPO training
- Explain how SmolLM2 achieves strong performance at 135M–1.7B parameters and what training decisions drive this
- Describe OLMo 2's fully open training stack and what it reveals about the relationship between data quality and model capability
- Identify at least three techniques from these papers that you could apply to improve postgres-sqlcoder-7b
- Synthesize across the three papers: what is the emerging consensus in 2024–2025 instruction tuning?

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

- 1.5h: Read Tulu 3 (focus: RLVR section and data ablations)
- 1.5h: Read SmolLM2 (focus: data curriculum and evaluation results)
- 1.5h: Read OLMo 2 (focus: mid-training data mixing and open infrastructure)
- 1.5h: Write synthesis notes in `reading_notes/week71_synthesis.md`

## Why this week matters

Three papers; one consensus: curate better data, verify rewards objectively, train in stages.
