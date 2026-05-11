# Week 19 Overview — Distributed Training: DDP, FSDP, and ZeRO

This file is your map for Week 19. Read it first; everything else fits inside it.

## The story this week

A single A100 80GB GPU can hold at most ~40B model parameters in FP16 (without optimizer states). A 7B model with AdamW optimizer states (2× params = 14B values) + gradients (7B values) + activations requires roughly 60–70GB — barely fits on one A100. A 70B model is impossible on a single GPU.

## What you need to do

- [ ] `pip install accelerate` (latest)
- [ ] `accelerate config` completed (choose "No distributed training" for Colab single GPU)
- [ ] A minimal nanoGPT training script (from Week 20 preview, or download from [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT))
- [ ] GitHub repo with `week-19-distributed-concepts/` directory

Concretely, by the end of the week you should be able to:

- Explain data parallelism, model parallelism, and pipeline parallelism at a conceptual level
- Describe ZeRO Stages 1, 2, and 3 and what each shards across GPUs
- Explain why FSDP superseded the original PyTorch DDP for large models
- Use HuggingFace Accelerate to run single-GPU training with minimal code changes
- Read a multi-GPU FSDP config and explain what each line does

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

- 1h: Read PyTorch DDP tutorial (understand the `init_process_group` and `all_reduce` concepts)
- 1.5h: Read ZeRO paper (Sections 1–4; focus on the memory analysis table in Section 2)
- 1h: Read HuggingFace Accelerate concept guides
- 2h: Coding — integrate Accelerate into your nanoGPT prototype and run it on Colab GPU
- 1h: Read the FSDP config for a publicly available training script (e.g., Llama 3's training config) and annotate each line in your `journal.md`
- 0.5h: Commit and write `journal.md` notes

## Why this week matters

**One-liner:** DDP replicates model; ZeRO shards optimizer/gradients/params; FSDP = ZeRO-3 in PyTorch; Accelerate makes it portable.
