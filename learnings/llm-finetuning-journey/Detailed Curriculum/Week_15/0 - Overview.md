# Week 15 Overview — From-Scratch GPT-2 124M Reproduction (Karpathy)

This file is your map for Week 15. Read it first; everything else fits inside it.

## The story this week

Every previous week has been building toward this: you understand attention (Weeks 9–10), you built a decoder-only model (Week 11), you added modern architecture components (Week 12), you implemented KV cache and sampling (Week 13), you read the LLaMA codebase (Week 14). Now you reproduce a real, published model from scratch.

## What you need to do

- [ ] Buy Colab Pro ($10, one month). Use the A100 runtime when available. This is a required spend.
- [ ] Install: `pip install tiktoken datasets wandb`
- [ ] W&B project `week-15-gpt2-repro`
- [ ] GitHub branch `week-15-gpt2-repro`
- [ ] 20GB+ free disk on Colab (for FineWeb-Edu or OpenWebText download)

Concretely, by the end of the week you should be able to:

- Reproduce GPT-2 124M training following Karpathy's methodology
- Implement mixed-precision training (torch.autocast with bfloat16)
- Implement gradient accumulation to simulate large batch sizes on limited hardware
- Understand Flash Attention and use `F.scaled_dot_product_attention` as a drop-in
- Explain what HellaSwag evaluation measures and why it's used during pretraining
- Achieve val loss within 5% of GPT-2 124M's published val loss on OpenWebText

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
| Watch Karpathy's full video (4h01m) — code along | 5 hrs |
| Run actual training on Colab Pro (A100) | 1 hr setup + monitor |
| Review W&B plots, compare to published numbers | 1 hr |
| Commit and document | 0.5 hrs |

This is the most time-intensive coding week in Phase 2. Spread the video across the full week. Do not rush it.

---

## Why this week matters

**This week in 15 words:** Reproduce GPT-2 124M with mixed precision, Flash Attention, and gradient accumulation — for real.
