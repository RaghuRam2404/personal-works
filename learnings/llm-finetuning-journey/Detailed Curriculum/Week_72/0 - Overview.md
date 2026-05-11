# Week 72 Overview — Frontier Reading 2: DeepSeek and Qwen Technical Reports

This file is your map for Week 72. Read it first; everything else fits inside it.

## The story this week

DeepSeek-V3 (DeepSeek-AI, 2024) is a 671B total parameter MoE model with only 37B active parameters per forward pass. Its significance for your work is not the scale — you cannot train or run it — but the architectural and training techniques it introduces that are now trickling into smaller open models.

## What you need to do

- [ ] Download PDFs: DeepSeek-V3 (arXiv 2412.19437), DeepSeek-R1 (arXiv 2501.12948), Qwen2.5 technical report (arXiv 2412.15115)
- [ ] `reading_notes/` directory from Week 71 available
- [ ] Your Week 75 candidates list ready: Llama 3.1 8B, Gemma 2 9B, DeepSeek-Coder-V2-Lite, DeepSeek-R1-Distill-Qwen-7B

Concretely, by the end of the week you should be able to:

- Explain the Mixture-of-Experts (MoE) architecture used in DeepSeek-V3 and why it enables frontier capability at lower active parameter counts
- Describe DeepSeek-R1's chain-of-thought reinforcement learning approach and relate it to your GRPO training
- Summarize Qwen2.5's training pipeline and identify the specific decisions relevant to SQL code generation
- Evaluate whether switching to a DeepSeek or Qwen base model for Week 75 is motivated by these papers
- Extract three concrete techniques to apply in your iteration weeks

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

- 1.5h: Read DeepSeek-V3 technical report (focus: MLA, MoE routing, training efficiency)
- 1.5h: Read DeepSeek-R1 (focus: cold start, GRPO, distillation to 7B)
- 1.0h: Read Qwen2.5 technical report (focus: data mixture, Coder variant, post-training)
- 1.5h: Write synthesis notes and base model decision matrix for Week 75
- 0.5h: Update `reading_notes/week72_synthesis.md`

## Why this week matters

R1 validates your GRPO; V3's MLA is the future of KV cache; Qwen2.5-Coder is a strong base and you already know why your DPO helped.
