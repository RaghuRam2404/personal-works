# Week 43 Overview — DPO: Skipping the Reward Model

This file is your map for Week 43. Read it first; everything else fits inside it.

## The story this week

The RLHF objective is:

## What you need to do

- [ ] Colab Pro (DPO on ultrafeedback_binarized with a small model fits in 16GB VRAM)
- [ ] Packages: `trl>=0.8.0`, `transformers>=4.38`, `datasets`, `peft`, `unsloth` (optional for speed)
- [ ] HuggingFace account — you will push the trained model
- [ ] Read the DPO paper Appendix A.1 before coding

Concretely, by the end of the week you should be able to:

- Derive the DPO loss from the KL-constrained RL objective in Appendix A.1 of the DPO paper
- Explain why DPO produces the same optimal policy as PPO-RLHF without running a RL training loop
- Identify the assumptions DPO makes and when they break down
- Run a DPO training job on `HuggingFaceH4/ultrafeedback_binarized` using TRL's DPO trainer
- Explain the relationship between the DPO loss and the Bradley-Terry reward model

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

- 2 hours: Read DPO paper fully, including Appendix A.1 derivation. Take notes on each step.
- 30 min: Re-read Week 42 notes on Bradley-Terry loss to connect the notation.
- 1 hour: Watch Umar Jamil's DPO explanation video (linked in Resources).
- 30 min: Read TRL DPO Trainer docs.
- 2.5–3 hours: Run philschmid's DPO notebook end-to-end on ultrafeedback_binarized.
- 30 min: Derive the DPO loss on paper without looking at notes. This is the acceptance criterion.

## Why this week matters

**One-liner:** DPO replaces PPO's training loop by reparameterizing the reward as a log-ratio of policy to reference model.
