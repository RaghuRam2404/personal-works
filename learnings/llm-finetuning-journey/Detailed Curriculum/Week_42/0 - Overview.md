# Week 42 Overview — PPO and the Original RLHF Stack

This file is your map for Week 42. Read it first; everything else fits inside it.

## The story this week

When you update policy parameters θ, you want to improve J(θ) but not overshoot. The Trust Region Policy Optimization (TRPO) paper addressed this with a KL constraint:

## What you need to do

- [ ] Clone TRL repo locally: `git clone https://github.com/huggingface/trl.git`
- [ ] Target file: `trl/trainer/ppo_trainer.py`
- [ ] No GPU required — this is a code-reading week
- [ ] Have the PPO paper ([arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)) and InstructGPT paper ([arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155)) open in a browser

Concretely, by the end of the week you should be able to:

- Derive the PPO clipping objective and explain why it replaces the raw policy gradient update
- Explain Generalized Advantage Estimation (GAE) and state its two hyperparameters (λ, γ)
- Describe the InstructGPT three-stage pipeline: SFT → reward model → RL fine-tuning
- Explain the role of the reference model and the KL penalty in RLHF
- Read and annotate TRL's PPOTrainer source code and identify where each component lives

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

- 2 hours: Read PPO paper (focus on sections 1–3). Read InstructGPT paper sections 3–4.
- 1 hour: Watch Yannic Kilcher's InstructGPT video.
- 30 min: Read Spinning Up GAE section.
- 2.5–3 hours: Read and annotate TRL PPOTrainer source code. Add inline comments explaining each step.
- 30 min: Write a 1-paragraph summary of what each stage of InstructGPT does and why.

## Why this week matters

**One-liner:** PPO clips the policy update ratio to prevent overshooting; RLHF adds a KL penalty to prevent reward hacking.
