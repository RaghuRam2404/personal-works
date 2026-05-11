# Week 30 Overview — LoRA: The Math and the Intuition

This file is your map for Week 30. Read it first; everything else fits inside it.

## The story this week

Full SFT on a 7B model requires storing and updating 7 billion parameters — plus gradients and optimizer states, which pushes memory requirements above 80GB. Even for a 0.5B model, full SFT is inefficient: most of those parameter updates are redundant.

## What you need to do

- [ ] PyTorch installed (Mac MPS or Colab Free T4)
- [ ] Your nanoGPT from Phase 2 available, or clone a reference implementation
- [ ] LoRA paper saved: https://arxiv.org/abs/2106.09685
- [ ] No new packages needed — pure PyTorch only this week

Concretely, by the end of the week you should be able to:

- Derive the LoRA weight update formula and count the trainable parameters for a given rank r
- Implement a `LoraLinear` module in PyTorch from scratch that passes a correctness test
- Explain why fine-tuning updates are intrinsically low-rank and what experimental evidence supports this
- Apply LoRA to a simple GPT-2 or nanoGPT model and verify it trains correctly
- Articulate the trade-off between rank r, parameter count, and model expressiveness

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
| Read LoRA paper fully (arxiv 2106.09685) | 2h |
| Watch Yannic Kilcher LoRA video (25m) + Raschka video (1h) | 1.5h |
| Implement `LoraLinear` from scratch | 1.5h |
| Apply LoRA to nanoGPT, verify trainable param count | 1h |
| Write blog-post-style writeup | 30m |
| Commit to GitHub | 15m |

## Why this week matters

**One-liner:** LoRA = frozen W + trainable low-rank delta_W = BA, using r×(d_in+d_out) params instead of d_in×d_out.
