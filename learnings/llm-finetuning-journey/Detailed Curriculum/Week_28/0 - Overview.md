# Week 28 Overview — What is Fine-Tuning, Really?

This file is your map for Week 28. Read it first; everything else fits inside it.

## The story this week

You finished Phase 3 having trained transformer language models from scratch and having understood scaling laws. Now the question changes: given an already powerful pretrained model, how do you specialize it?

## What you need to do

- [ ] GitHub repo open and accessible (your Phase 3 repo is fine)
- [ ] Excalidraw account (free at excalidraw.com) OR pen/paper + camera
- [ ] InstructGPT paper downloaded: https://arxiv.org/abs/2203.02155
- [ ] Karpathy video queued: https://www.youtube.com/watch?v=7xTGNNLPyMI
- [ ] No GPU needed this week

Concretely, by the end of the week you should be able to:

- Distinguish continued pretraining, supervised fine-tuning (SFT), and instruction tuning — and explain when each is appropriate
- Describe the modern post-training pipeline: SFT → DPO → GRPO
- Explain why fine-tuning on a small labeled dataset can dramatically shift model behavior without destroying pretrained knowledge
- Read and understand InstructGPT's training methodology (sections 1–3)
- Map Karpathy's conceptual walkthrough of LLMs to the technical concepts you have built so far

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

| Activity | Time |
|---|---|
| Watch Karpathy's "Deep Dive into LLMs like ChatGPT" (3h31m) | 3.5h |
| Read InstructGPT paper sections 1–3 | 1h |
| Read HuggingFace LLM Course Chapter 11 intro | 30m |
| Read Karpathy's 2025 Year in Review blog post | 30m |
| Draw the post-training pipeline diagram | 30m |
| Commit diagram to GitHub | 15m |

## Why this week matters

**One-liner:** Three ways to update a pretrained model; SFT is the first required stage for any task-specific deployment.
