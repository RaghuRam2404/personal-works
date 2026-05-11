# Week 46 Overview — GRPO and RLVR: The 2025 Breakthrough

This file is your map for Week 46. Read it first; everything else fits inside it.

## The story this week

PPO was designed for dense reward environments. For LLMs with verifiable rewards, the problems are:
1. You need a critic network (value function) that estimates V(s_t) at each token position. This critic is expensive and hard to train — it receives a signal only at the end of each episode (sparse).
2. The actor and critic together require 2× the memory of a single model.
3. GAE requires the critic to be accurate before it helps. Early in training, the critic is random noise, and GAE advantage estimates are useless.
4. Running the actor + critic + reward model + reference model for every training step is prohibitively expensive for 7B+ models.

## What you need to do

- [ ] Clone TRL: `git clone https://github.com/huggingface/trl.git`
- [ ] Target file: `trl/trainer/grpo_trainer.py`
- [ ] Papers open: DeepSeekMath (arxiv.org/abs/2402.03300), DeepSeek-R1 (arxiv.org/abs/2501.12948)
- [ ] No GPU needed this week — reading and writing only

Concretely, by the end of the week you should be able to:

- Explain GRPO (Group Relative Policy Optimization) and why it eliminates the critic network
- State why verifiable rewards (RLVR) changed the alignment landscape in 2025
- Describe how DeepSeek-R1 applied GRPO at scale to produce reasoning behavior
- Read and annotate TRL's GRPOTrainer source code
- Write a 2-page explainer in your own words connecting GRPO theory to your SQL training plan

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

- 2 hours: Read DeepSeekMath paper (Section 3 on GRPO) and DeepSeek-R1 paper (Sections 2–3).
- 1 hour: Read HuggingFace LLM Course Chapter 12 on implementing GRPO.
- 1 hour: Watch Karpathy's LLM Year in Review (the RLVR section) and Yannic Kilcher's DeepSeek-R1 video.
- 2–3 hours: Read and annotate TRL GRPOTrainer source.
- 30 min: Write your 2-page explainer in `week-46-grpo/grpo_explainer.md`.

## Why this week matters

**One-liner:** GRPO replaces the PPO critic with within-group reward normalization; verifiable rewards make the group mean an exact baseline.
