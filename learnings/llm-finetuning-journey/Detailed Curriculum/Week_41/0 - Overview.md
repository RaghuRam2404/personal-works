# Week 41 Overview — RL Primer: The Minimum You Need for LLMs

This file is your map for Week 41. Read it first; everything else fits inside it.

## The story this week

An MDP is a 5-tuple: (S, A, P, R, γ).

## What you need to do

- [ ] Google Colab (Free tier is sufficient; CartPole is CPU-only)
- [ ] Python packages: `torch`, `gymnasium` (the maintained fork of `gym`)
- [ ] No GPU needed — CartPole environment runs on CPU in seconds

Concretely, by the end of the week you should be able to:

- Define an MDP (Markov Decision Process) and identify its components in a language model setting
- State the policy gradient theorem and derive the REINFORCE update rule from it
- Explain the difference between on-policy and off-policy learning and why it matters for LLM training
- Implement a working REINFORCE agent that solves CartPole-v1 in PyTorch
- Map RL terminology (reward, policy, episode) onto the RLHF pipeline for language models

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

- 2 hours: Read HuggingFace Deep RL Course Units 1, 2, 4, 8 (linked in Resources). Take notes.
- 1 hour: Read Lilian Weng's policy gradient post. Focus on REINFORCE and baseline subtraction.
- 1 hour: Read Spinning Up intro page. Make sure you can state the policy gradient theorem from memory.
- 2–3 hours: Implement CartPole REINFORCE in PyTorch (the Assignment).
- 30 min: Write your own derivation of the policy gradient on paper. Do not look at the notes. This is the acceptance criterion.

## Why this week matters

**One-liner:** The policy gradient theorem lets you optimize non-differentiable rewards via log-probability weighting.
