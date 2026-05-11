# Week 20 Overview — Pretraining Setup: Codebase, Model Size, and Tokenizer

This file is your map for Week 20. Read it first; everything else fits inside it.

## The story this week

You will build on [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT), but you should not simply copy-paste it. The goal is to understand every line. A minimal pretraining codebase needs:

## What you need to do

- [ ] `pip install tokenizers datasets torch accelerate wandb numpy`
- [ ] W&B account created and `wandb login` completed
- [ ] Colab Pro session with A100 runtime (needed for Week 21; verify access now)
- [ ] GitHub repo with `pretrain-50m/` directory
- [ ] HuggingFace account for dataset access

Concretely, by the end of the week you should be able to:

- Set up a clean pretraining codebase based on nanoGPT
- Justify a ~50M parameter GPT configuration mathematically
- Train a Byte-Pair Encoding (BPE) tokenizer from scratch using HuggingFace tokenizers
- Build a data pipeline that streams FineWeb-Edu and converts it to tokenized `.bin` files
- Verify that your codebase is wired up correctly with a short sanity-check training run

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

- 1h: Fork nanoGPT, study the model definition, write your own version
- 1h: Train BPE tokenizer on FineWeb-Edu sample
- 1.5h: Write data pipeline (prepare_data script + TokenDataset)
- 2h: Write training loop with Accelerate + W&B logging
- 1h: Run sanity checks, verify loss starts at ~log(vocab_size)
- 0.5h: Commit everything with `week-20-pretrain-setup`

## Why this week matters

**One-liner:** 8 layers × d_model=768 × 12 heads = ~57M params; BPE tokenizer; memory-mapped .bin files; sanity-check loss ≈ log(vocab_size) before training.
