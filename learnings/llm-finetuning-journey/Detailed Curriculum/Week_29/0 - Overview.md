# Week 29 Overview — Full SFT on a Tiny Model

This file is your map for Week 29. Read it first; everything else fits inside it.

## The story this week

`SFTTrainer` is built on top of HuggingFace `Trainer` and adds two key conveniences: (1) automatic input masking — it masks the prompt tokens from the loss so only response tokens contribute, and (2) packing — it can concatenate multiple short examples into a single sequence up to `max_seq_length`, improving GPU utilization.

## What you need to do

- [ ] Colab Pro active (needed for A100/T4 GPU)
- [ ] HuggingFace account with write token (for model push)
- [ ] Weights & Biases account (free tier is fine)
- [ ] Packages: `pip install trl transformers datasets peft accelerate wandb`
- [ ] Dataset: [sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context) or Spider — load 1K rows for training, 100 rows held out for evaluation

Concretely, by the end of the week you should be able to:

- Set up a complete SFT training loop using HuggingFace `SFTTrainer` from `trl`
- Format a PostgreSQL text-to-SQL dataset into a chat-template-compatible format
- Configure and debug `Qwen2.5-0.5B`'s tokenizer, including the chat template
- Push a fine-tuned model to HuggingFace Hub and log training to Weights & Biases
- Interpret training loss curves and identify basic failure modes

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
| Read SFTTrainer docs + HuggingFace fine-tuning tutorial | 1h |
| Set up Colab Pro notebook, install packages | 30m |
| Download and format 1K SQL dataset | 1h |
| Write and debug the SFT training script | 2h |
| Run training, monitor W&B | 1h |
| Push model to HuggingFace Hub, commit to GitHub | 30m |

## Why this week matters

**One-liner:** SFTTrainer + chat template + 1K SQL pairs = your first working SQL fine-tune in under 15 minutes.
