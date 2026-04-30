# Week 46 Assignment — Annotate GRPOTrainer and Write GRPO Explainer

## Setup Checklist

- [ ] Clone TRL: `git clone https://github.com/huggingface/trl.git`
- [ ] Target file: `trl/trainer/grpo_trainer.py`
- [ ] Papers open: DeepSeekMath (arxiv.org/abs/2402.03300), DeepSeek-R1 (arxiv.org/abs/2501.12948)
- [ ] No GPU needed this week — reading and writing only

---

## Task 1 — Annotate TRL GRPOTrainer

**Goal:** Produce an annotated version of `grpo_trainer.py` that connects every key step to the algorithm in DeepSeekMath Section 3.

**Requirements:**
- Find and annotate 5 key locations with 3–5 lines each:
  1. Where K completions are generated per prompt
  2. Where rewards are computed for each completion (the `reward_fn` call)
  3. Where group-relative normalization (mean/std) is applied to compute advantages
  4. Where the PPO clip is applied using those advantages
  5. Where the KL divergence penalty is computed and applied

- For location 3, also write: "This is equivalent to REINFORCE baseline subtraction from Week 41, applied at the group level instead of the episode level."
- Create `week-46-grpo/grpo_trainer_annotated.py`

**Deliverable:** `week-46-grpo/grpo_trainer_annotated.py`. GitHub commit: `week-46-grpo-annotated`.

**Hints:**
- Search for `std` and `mean` in the file — the normalization is near them
- The `generate` call is usually in a method called `_generate_completions` or similar
- The reward function may be called `compute_rewards` or via a `reward_model` attribute

---

## Task 2 — Write a 2-Page GRPO Explainer

**Goal:** Write a clear, technical explanation of GRPO in your own words, without copying the paper.

**Requirements:**
- File: `week-46-grpo/grpo_explainer.md`
- Length: 500–700 words (2 pages)
- Must cover:
  1. What problem GRPO solves (why PPO has a critic, why that is expensive, why GRPO eliminates it)
  2. The group-relative advantage formula and what it means intuitively
  3. How GRPO connects to REINFORCE baseline subtraction (Week 41)
  4. Why GRPO is better than DPO for verifiable rewards
  5. One concrete SQL example: prompt = "Get total revenue by product", K=4 completions with rewards [1, 0, 1, 0], computed advantages
- You must write this in your own words — no copy-paste from papers

**Deliverable:** `week-46-grpo/grpo_explainer.md`

---

## Task 3 — Compare PPO, DPO, and GRPO

**Goal:** Build a comparison table for reference in Weeks 47–52.

**Requirements:**
- Create `week-46-grpo/algorithm_comparison.md`
- Table with rows: PPO, DPO, GRPO; columns: requires_critic, requires_RM, on_or_off_policy, best_for, memory_cost, key_hyperparams
- Write one paragraph below the table: "For my SQL domain, I will use GRPO because..."

**Deliverable:** `week-46-grpo/algorithm_comparison.md`

---

## Stretch Goals

- Read DeepSeek-R1's "cold start" section (Section 3.1). How does it differ from standard GRPO? Could you apply cold start to your SQL model?
- Find the `num_generations` hyperparameter in TRL's GRPOConfig and check what the default K is. Is it 8? 16? Something else?
- Implement GRPO's group-relative normalization from scratch in 10 lines of PyTorch and verify: rewards = [1, 0, 1, 0] → advantages = [0.577, -0.577, 0.577, -0.577] (std normalization).
