# Week 9 Overview — The Original Attention Mechanism (Bahdanau, 2014)

This file is your map for Week 9. Read it first; everything else fits inside it.

## The story this week

In 2014, the dominant architecture for machine translation was sequence-to-sequence (seq2seq) with an encoder RNN and a decoder RNN. The encoder processes the input sequence one token at a time and compresses the entire source sentence into a single fixed-size vector — the context vector. The decoder then uses that vector to generate the target sequence.

## What you need to do

- [ ] PyTorch installed (any version ≥ 2.0)
- [ ] `matplotlib` installed for attention heatmap plotting
- [ ] GitHub repo initialized with a `week-09-bahdanau-attn` branch
- [ ] W&B account set up (free tier) — project name `week-09-bahdanau-attn`

Concretely, by the end of the week you should be able to:

- Explain why attention was invented and what problem in seq2seq models it solved
- Derive the Bahdanau additive attention score function on paper
- Implement a seq2seq LSTM with Bahdanau attention in PyTorch from scratch
- Visualize attention weight matrices and interpret what the model "looks at"
- Distinguish additive (Bahdanau) from multiplicative (Luong) attention
- Describe the alignment mechanism and why it enables translation of long sentences

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
| Read Bahdanau paper (1409.0473) with notes | 2.5 hrs |
| Read Lilian Weng blog post | 0.5 hrs |
| Watch Yannic Kilcher intro (first 10 min) | 0.25 hrs |
| Implement seq2seq + attention in PyTorch | 2.5 hrs |
| Plot attention heatmap, write commit + notes | 0.75 hrs |

---

## Why this week matters

**This week in 15 words:** Attention lets the decoder dynamically query all encoder states instead of one fixed vector.
