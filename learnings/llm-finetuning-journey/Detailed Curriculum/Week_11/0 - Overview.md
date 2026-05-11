# Week 11 Overview — Decoder-Only Transformers and the GPT Family

This file is your map for Week 11. Read it first; everything else fits inside it.

## The story this week

The original Transformer (Week 10) was designed for translation — a task with a clear source and target sequence. But language modeling — predicting the next token given a context — doesn't need an encoder. There's no separate "source" to compress. The input is the context itself, and the model autoregressively predicts what comes next.

## What you need to do

- [ ] Colab Free (this fits comfortably for the Shakespeare run; the SQL run may need a GPU)
- [ ] W&B project `week-11-nanogpt`
- [ ] GitHub branch `week-11-nanogpt`
- [ ] Watch the full Karpathy video (1h56m) BEFORE coding — code along during the video

Concretely, by the end of the week you should be able to:

- Explain why GPT dropped the encoder and what was gained by doing so
- Describe causal language modeling as a pretraining objective and why it requires no labeled data
- Implement a decoder-only transformer (nanoGPT style) from scratch in PyTorch
- Explain weight tying between the embedding table and the LM head
- Describe the residual stream view of transformer computation
- Train nanoGPT on Tiny Shakespeare and on a SQL corpus; compare generated samples

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
| Read GPT-1 and GPT-2 papers (focus on architecture sections) | 1 hr |
| Skim GPT-3 architecture + scaling section | 0.5 hrs |
| Watch Karpathy nanoGPT video (1h56m), code along | 4 hrs |
| Retrain on SQL corpus, generate samples, commit | 1.5 hrs |

---

## Why this week matters

**This week in 15 words:** Drop the encoder; predict next tokens autoregressively; weight-tied embeddings; causal mask enables parallelism during training.
