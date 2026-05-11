# Week 50 Assignment — Iteration Run 1

## Setup Checklist

- [ ] Week 48 eval report available (v1/v2/v3 comparison)
- [ ] v3 model checkpoint accessible (local or HF Hub)
- [ ] Reward function from Week 47 available
- [ ] RunPod account ready (or Colab Pro for smaller experiments)
- [ ] Week 48 W&B run logs accessible

---

## Task 1 — Diagnosis and Hypothesis

**Goal:** Before running anything, diagnose your v3 model's failures with evidence.

**Requirements:**
- Open your Week 48 eval report
- Identify the top 2 failure modes from the list in the curriculum (or identify a new one with evidence)
- For each failure mode, write:
  - Evidence: "Complex query accuracy is 48% for v3 vs 48% for v2 — no improvement."
  - Hypothesis: "I believe this is because..."
  - Proposed fix: specific change to reward function, data, or config
  - Expected metric change: "Complex query accuracy should improve to 58%+ after this fix."
- Rank the two failure modes by expected impact. You will tackle the higher-impact one this week.

**Deliverable:** `week-50-iteration/diagnosis.md`

---

## Task 2 — Implement the Fix

**Goal:** Make the specific change identified in Task 1.

**Requirements (choose the path that matches your diagnosis):**

**Path A — Reward function fix:**
- Modify `reward_fn.py` based on the diagnosis
- Re-run the Week 47 diagnostic test (100 prompts × 4 completions) with the new reward function
- Compare the reward distribution: is it more spread out than before?
- Document: `week-50-iteration/reward_fn_v2.py` and `reward_diagnostics_v2.md`

**Path B — Dataset expansion:**
- Generate 200–300 new SQL prompts for the failing query type
- Label using your execution harness
- Add to the existing training set
- Verify: new prompts have v3 success rate of 20–60% (using diagnostic test)
- Document: `week-50-iteration/prompt_expansion.jsonl` and `expansion_diagnostics.md`

**Path C — Hyperparameter change:**
- Identify the specific hyperparameter to change (temperature, β, K, learning rate)
- Document why you are changing it (reference the curriculum's failure mode descriptions)
- Prepare the modified training config
- Document: `week-50-iteration/grpo_config_v2.py`

You may combine paths (e.g., Path A + Path B), but do not make more than 2 changes simultaneously — you need to isolate which change drove any improvement.

**Deliverable:** Files from the path(s) you chose

---

## Task 3 — Run the Targeted Experiment

**Goal:** Run a targeted GRPO experiment (200–500 steps) from the v3 checkpoint.

**Requirements:**
- Starting checkpoint: v3 (not v1 or v2 — you are patching v3, not starting over)
- Steps: 300 (adjust based on the severity of the fix — more complex fixes need more steps)
- Monitor: the same W&B metrics as Week 48
- Checkpoint every 50 steps
- After training: run the full eval pipeline on the new model

**Deliverable:**
- New model pushed: `<your-handle>/postgres-sqlcoder-7b-v3-iter1`
- `week-50-iteration/training_log.md` with W&B link

---

## Task 4 — Evaluate and Log

**Goal:** Compare v3 and v3-iter1 on the held-out test set.

**Requirements:**
- Run full eval on v3 and v3-iter1 (same 200-query test set as always)
- Report: did the targeted metric improve by the expected amount?
- Write the full iteration log: Hypothesis → Experiment → Result → Analysis
- Decide: should Week 51 continue this fix or try the second failure mode?

**Deliverable:** `week-50-iteration/iteration_log.md`

---

## Stretch Goals

- If your targeted fix improved the metric: try a second round with a stronger version (e.g., 300 more complex prompts if 100 helped, or β=0.10 if 0.15 helped).
- If the fix did not help: run a controlled ablation — revert only the reward function change and see if keeping the dataset expansion alone helps.
- Read DAPO's "clip-higher" trick and implement it: raise the upper clip bound from 1+ε to 1+3ε for completions with positive advantage. This can help when the model needs large probability increases to escape local minima.
