# Week 10 Overview — "Attention Is All You Need" — The Paper, Line by Line

This file is your map for Week 10. Read it first; everything else fits inside it.

## The story this week

The 2017 paper by Vaswani et al. proposed dropping RNNs entirely. Recurrence requires sequential processing — you cannot compute hidden state `h_t` until `h_{t-1}` is ready, which limits parallelism during training. The Transformer replaces recurrence with self-attention: every position in the sequence can attend to every other position in a single matrix operation. This unlocks massive parallelism and enables training on far larger datasets.

## What you need to do

- [ ] PyTorch ≥ 2.0
- [ ] `torchtext` or manual data loading (the Annotated Transformer uses raw Python)
- [ ] Colab Free tier (encoder-decoder Transformer on a toy task fits in free RAM)
- [ ] GitHub branch `week-10-annotated-transformer`
- [ ] W&B project `week-10-transformer`

Concretely, by the end of the week you should be able to:

- Derive scaled dot-product attention on paper and explain the `1/sqrt(d_k)` scaling term
- Describe multi-head attention: why multiple heads, what each head sees, how outputs are concatenated
- Explain sinusoidal positional encoding and why it allows generalization to unseen lengths
- Implement the complete encoder-decoder Transformer from The Annotated Transformer
- Explain the role of each sublayer (self-attention, cross-attention, FFN) in the encoder and decoder
- Identify the key differences between the 2017 Transformer and RNN-based seq2seq

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
| Read "Attention Is All You Need" (first pass, cover-to-cover) | 1.5 hrs |
| Read The Annotated Transformer (nlp.seas.harvard.edu) | 1 hr |
| Watch Yannic Kilcher full video (28 min) + 3B1B videos (~53 min total) | 1.5 hrs |
| Type out Annotated Transformer implementation | 2.5 hrs |
| Train on toy task, write notes, commit | 1 hr |

---

## Why this week matters

**This week in 15 words:** Self-attention replaces recurrence; parallelism unlocks scale; every architectural choice has a mathematical reason.
