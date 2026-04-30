# Week 42 Assignment — Annotate TRL PPOTrainer and Map InstructGPT

## Setup Checklist

- [ ] Clone TRL repo locally: `git clone https://github.com/huggingface/trl.git`
- [ ] Target file: `trl/trainer/ppo_trainer.py`
- [ ] No GPU required — this is a code-reading week
- [ ] Have the PPO paper ([arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)) and InstructGPT paper ([arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155)) open in a browser

---

## Task 1 — Annotate TRL PPOTrainer

**Goal:** Produce an annotated version of `ppo_trainer.py` that a newcomer could read to understand RLHF without the papers.

**Requirements:**
- Find and annotate the following 6 locations with inline comments. For each, write 3–5 lines explaining what is happening and why:
  1. Where the reference model is called to compute KL divergence
  2. Where the reward from the reward model is combined with the KL penalty
  3. Where GAE (advantage estimation) is computed
  4. Where the PPO clipping loss is computed
  5. Where the value function (critic) loss is computed
  6. Where completions are generated from the current policy (actor)
- Create `week-42-ppo/ppo_trainer_annotated.py` — a copy of the file with your comments inserted

**Deliverable:** `week-42-ppo/ppo_trainer_annotated.py`. GitHub commit: `week-42-ppo-annotated`.

**Hints:**
- Search for `kl_divergence` or `kl_ctl` in the file — the KL penalty computation is nearby
- The `compute_advantages` method name may vary by TRL version; search for `gae` or `advantages`
- GAE is computed as a reversed cumulative sum; look for a loop that iterates backward over timesteps
- The clipping loss is literally `torch.clamp(ratio, 1-self.cliprange, 1+self.cliprange)` or similar

---

## Task 2 — InstructGPT Pipeline Diagram

**Goal:** Draw (or write in Markdown table form) the full InstructGPT three-stage pipeline with concrete details filled in for the SQL domain.

**Requirements:**
- Three columns: Stage, What Happens, SQL-Domain Equivalent
- Fill in the SQL equivalent for each stage:
  - SFT: what data? (your Week 44/45 datasets)
  - Reward Model: what does a "preferred" SQL look like vs. "rejected"?
  - RL/PPO: what is the reward signal for a SQL generation task?
- Include a row for "KL Penalty" explaining why it is needed in the SQL context specifically (hint: without it, the model could generate SQL that always triggers a Postgres error that the naive reward model scores highly)

**Deliverable:** `week-42-ppo/instructgpt_sql_pipeline.md`

---

## Task 3 — GAE Implementation

**Goal:** Implement GAE from scratch in 20 lines of PyTorch.

**Requirements:**
- Input: `rewards` (list of floats), `values` (tensor of V(s_t) estimates), `gamma=0.99`, `lam=0.95`
- Output: `advantages` tensor and `returns` tensor
- Do not use any library functions for GAE — implement the backward loop yourself
- Add a unit test: for a single-step episode with reward=1.0 and V(s)=0.5, verify A = 1.0 + 0 − 0.5 = 0.5 (with γ=1, λ=1)

**Deliverable:** `week-42-ppo/gae.py`

---

## Stretch Goals

- Find TRL's default values for `cliprange` (ε), `vf_coef` (c1), `ent_coef` (c2), and `kl_coef` (β). Compare to the values used in the original InstructGPT paper. Are they the same?
- Implement a minimal PPO training loop on CartPole (extending Week 41's REINFORCE). Replace the REINFORCE update with the PPO clipping loss and add a value network.
- Read the [37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) and note which 5 details you think matter most for LLM fine-tuning specifically.
