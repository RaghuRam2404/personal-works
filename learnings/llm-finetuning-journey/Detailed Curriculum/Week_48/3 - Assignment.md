# Week 48 Assignment — Run GRPO on RunPod A100

## Setup Checklist

- [ ] RunPod account with payment method (A100 80GB recommended; 40GB works with 4-bit)
- [ ] RunPod pod started: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` image
- [ ] Week 47 training script (`grpo_train.py`) uploaded to RunPod via SSH/scp
- [ ] Week 47 reward function (`reward_fn.py`) uploaded to RunPod
- [ ] W&B API key set: `export WANDB_API_KEY=...`
- [ ] HF token set: `huggingface-cli login`
- [ ] Your training prompt dataset with reference SQL loaded on RunPod

---

## Task 1 — One-Step Verification

**Goal:** Before starting the full run, verify the training script completes exactly 1 step without error.

**Requirements:**
- Modify `grpo_train.py` to set `max_steps=1` (temporarily)
- Run: `python grpo_train.py`
- Verify: the step completes, reward values are logged, no NaN or OOM error
- Record: memory usage with `nvidia-smi` during the step
- If OOM: reduce K to 4 or enable 4-bit quantization; document the change

**Deliverable:** `week-48-grpo/one_step_log.txt` — paste of the terminal output from step 1

---

## Task 2 — Full GRPO Training Run

**Goal:** Train postgres-sqlcoder-7b-v3-grpo with your reward function.

**Requirements:**
- Config:
  - num_generations (K): 8 (or 4 if A100 40GB is memory-constrained)
  - max_completion_length: 256
  - learning_rate: 5e-7
  - max_steps: 1000
  - save_steps: 50
  - W&B project: `week-48-grpo-sql`
  - W&B run name: `grpo-v3-run1`
- Monitoring requirements (check every 30 minutes):
  - `mean_reward` trending upward (even slowly)
  - KL divergence < 10 nats
  - Loss not NaN
  - `reward_std` not consistently 0
- If any red flag triggers: stop, diagnose, apply intervention from the curriculum
- Run for full 1000 steps (6–10 hours on A100 80GB)

**Deliverable:**
- Pushed model: `<your-handle>/postgres-sqlcoder-7b-v3-grpo`
- W&B run link in `week-48-grpo/training_log.md`
- Checkpoint saved locally at steps 500 and 1000

---

## Task 3 — Evaluate v1 vs. v2 vs. v3

**Goal:** Produce the definitive three-way comparison on the held-out test set.

**Requirements:**
- Use your 200-query held-out test set
- Run all three models (v1, v2, v3) on all 200 prompts with greedy decoding
- For each model, compute:
  - Execution accuracy
  - Semantic accuracy  
  - Syntax error rate
  - Complex query execution accuracy (top 50 hardest prompts)
  - Mean generation length (tokens)
- Produce a Markdown table with all metrics
- Add one paragraph of analysis: "v3 improved over v2 on X because... it did not improve on Y because..."

**Acceptance criterion:** v3 execution accuracy > v2 execution accuracy by ≥ 5pp.

**Deliverable:** `week-48-grpo/eval_report_v1_v2_v3.md`

---

## Task 4 — Training Run Analysis

**Goal:** Analyze what the GRPO training learned from the W&B logs.

**Requirements:**
- Download W&B run data as CSV
- Plot (using matplotlib or pandas): mean_reward vs. step, reward_std vs. step, KL vs. step
- Identify: at which step did mean_reward first significantly improve? What happened around that step?
- Note: did reward_std decrease over training (suggesting the model is collapsing toward a single solution)?

**Deliverable:** `week-48-grpo/training_analysis.md` with plots as embedded images

---

## Stretch Goals

- Run a second GRPO experiment with K=16 (if budget allows). Compare final mean_reward, convergence speed, and eval accuracy vs. K=8.
- After GRPO training, manually generate 10 responses from v3 for hard prompts (3+ JOIN, CTEs). Does v3 generate chain-of-thought reasoning before the SQL? (This is the DeepSeek-R1 "aha moment" in your domain.)
- Check the average completion length for v3 vs. v2. Did GRPO make the model more or less verbose? Why?
