# Week 2 Overview — MLPs, Activations, Initialization, and the Bag of Tricks

This file is your map for Week 2. Read it first; everything else fits inside it.

## The story this week

An MLP is a stack of linear transformations alternated with nonlinear activations. For a single layer:

## What you need to do

- [ ] W&B is configured and `wandb login` works. Create a project called `week-02-makemore`.
- [ ] Spider dataset downloaded: `git clone https://github.com/taoyds/spider.git` or download from [yale-lily/spider](https://github.com/taoyds/spider). You only need the JSON files — no database access required this week.
- [ ] Karpathy's makemore videos queued: Parts 1, 2, and 3.
- [ ] Folder `week_02/` created in your `llm-finetuning-journey` repo.

Concretely, by the end of the week you should be able to:

- Build a multi-layer perceptron in PyTorch using both `nn.Parameter` (manual) and `nn.Linear` (idiomatic), and explain the difference.
- Choose between ReLU, GELU, and Tanh activations for a given situation and justify the choice.
- Derive Kaiming (He) initialization on paper, and explain why it is the default for ReLU networks.
- Implement batch normalization from scratch and explain what it computes during training vs. inference.
- Diagnose overfitting from a W&B loss curve and apply dropout as a mitigation.
- Train a bigram and MLP language model on the makemore names dataset, then swap it to a SQL keyword dataset.

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
| Read Karpathy "Yes you should understand backprop" | 30 min |
| Read Deep Learning Book Ch. 6 (activations + init sections) | 1 h |
| Watch makemore Part 1 (bigram LM) — code along | 1.25 h |
| Watch makemore Part 2 (MLP LM) — code along | 1.25 h |
| Watch makemore Part 3 (Activations, Gradients, BatchNorm) — code along | 2 h |
| Swap dataset to SQL keywords, train, log to W&B | 1 h |
| Write journal entry and commit | 30 min |

## Why this week matters

**This week in 15 words:** MLP = Linear + activation; init and batch norm control whether gradients flow cleanly.
