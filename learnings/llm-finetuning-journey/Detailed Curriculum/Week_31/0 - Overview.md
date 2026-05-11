# Week 31 Overview — LoRA via the `peft` Library, Target Modules, and Rank Sweeps

This file is your map for Week 31. Read it first; everything else fits inside it.

## The story this week

The `peft` library (Parameter-Efficient Fine-Tuning, by HuggingFace) wraps the math you implemented in Week 30 into a clean API. It handles:
- Replacing target linear layers with LoRA-augmented versions automatically
- Freezing non-adapter parameters
- Saving and loading only the adapter weights (small files: 10–100MB vs. the full 14GB model)
- Merging adapters for inference

## What you need to do

- [ ] Colab Pro active (A100 or T4)
- [ ] `pip install peft trl transformers datasets accelerate wandb`
- [ ] 5K SQL training examples formatted from Week 29 (reuse your dataset)
- [ ] W&B project `week-31-lora-sweep` created
- [ ] HuggingFace write token set

Concretely, by the end of the week you should be able to:

- Use `peft` to apply LoRA to any HuggingFace model in under 20 lines of code
- Enumerate the target_modules for Qwen2.5-Coder-1.5B and justify which ones to include
- Run a W&B sweep over rank r ∈ {8, 16, 32, 64} and interpret the resulting loss curves
- Explain the practical trade-off between rank, parameter count, training speed, and generalization on small datasets
- Identify when a LoRA rank is too high for a given dataset size

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
| Read PEFT docs + Raschka's article on target_modules | 1h |
| Set up Colab Pro, install peft, format 5K dataset | 1h |
| Write LoRA fine-tuning script with peft + SFTTrainer | 1.5h |
| Run rank sweep (4 runs) — can run in parallel if 2 GPUs available | 2h |
| Analyze W&B sweep results, write comparison report | 1h |
| Commit to GitHub | 30m |

## Why this week matters

**One-liner:** peft wraps Week 30's LoRA math; enumerate target_modules explicitly, set alpha=2r, sweep rank 8→64 to find the overfitting boundary.
