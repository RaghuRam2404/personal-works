# Week 1 Overview — PyTorch Tensors, Autograd, and the Training Loop

This file is your map for Week 1. Read it first; everything else fits inside it.

## The story this week

A tensor is a multi-dimensional array with a fixed dtype living on a device (CPU, CUDA, or MPS). PyTorch tensors are the unit of computation: every input, every weight, every gradient is a tensor.

## What you need to do

Before starting, verify the following:

Concretely, by the end of the week you should be able to:

- Create, manipulate, and index PyTorch tensors confidently, including broadcasting rules.
- Explain what a computational graph is and how PyTorch builds one dynamically.
- Trace `loss.backward()` through a small network by hand, matching it to PyTorch's computed gradients.
- Write the canonical PyTorch training loop from memory without referring to documentation.
- Identify and fix three common training-loop bugs: missing `optimizer.zero_grad()`, forgetting `model.eval()` during validation, and shapes mismatches in the loss function.
- Re-implement Karpathy's `micrograd` engine to prove you understand autodiff from first principles.

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
| Read PyTorch basics tutorial (all 8 sections) | 1.5 h |
| Watch Karpathy micrograd video (2h25m) — code along | 2.5 h |
| Implement micrograd from scratch | 1 h |
| Port your old Python NN to PyTorch | 1 h |
| Write `journal.md`, commit to GitHub | 30 min |
| Read Karpathy's "Recipe for Training NNs" | 30 min |

## Why this week matters

**This week in 15 words:** PyTorch builds a dynamic compute graph; `backward()` chains gradients; the training loop has 5 steps.
