# Week 4 Overview — RNNs, LSTMs, and Why We Abandoned Them

This file is your map for Week 4. Read it first; everything else fits inside it.

## The story this week

An RNN processes sequences one step at a time, maintaining a hidden state `h_t` that summarizes all prior inputs:

## What you need to do

- [ ] Spider SQL corpus from Week 2 available: `week_02/sql_corpus.txt`. If not, re-run `extract_sql_tokens.py`.
- [ ] For character-level LSTM, you need the raw SQL query strings (not token sequences). Extract them: load `train_spider.json`, take the `query` field, concatenate with newlines → `week_04/sql_queries.txt`.
- [ ] Colab Free tier available. Training time: ~30 minutes on T4.
- [ ] W&B project `week-04-char-lstm` created.

Concretely, by the end of the week you should be able to:

- Implement a vanilla RNN and an LSTM cell from scratch in PyTorch using only tensor operations.
- Explain the vanishing gradient problem in RNNs mathematically, and describe how LSTM gates address it.
- Train a character-level LSTM on a SQL corpus and generate plausible-looking SQL character sequences.
- Explain truncated backpropagation through time (TBPTT) and implement it with `detach()`.
- Articulate precisely why LSTMs were superseded by transformers: parallelization and long-range dependency handling.
- Use teacher forcing correctly during training and understand why it diverges from autoregressive inference.

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
| Read Colah's LSTM blog post | 1 h |
| Read Karpathy's "Unreasonable Effectiveness of RNNs" | 30 min |
| Watch StatQuest RNN video (16m) | 20 min |
| Watch StatQuest LSTM video (20m) | 25 min |
| Implement RNN cell from scratch | 1 h |
| Implement LSTM cell from scratch | 1 h |
| Train char-LSTM on SQL corpus, generate samples | 2 h |
| Journal + commit | 30 min |

## Why this week matters

**This week in 15 words:** LSTM gates solve vanishing gradients; parallelism and attention are why transformers won.
