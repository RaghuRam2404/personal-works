# Week 49 Assignment — Build the Alignment Method Comparison

## Setup Checklist

- [ ] Papers open: KTO (2402.01306), ORPO (2403.07691), SimPO (2405.14734)
- [ ] Phase 5 notes from Weeks 41–48 available for reference
- [ ] No GPU required this week

---

## Task 1 — Build the Full Comparison Table

**Goal:** Produce a single reference table covering all 6 alignment methods from Phase 5.

**Requirements:**
- File: `week-49-zoo/alignment_comparison.md`
- Rows: PPO, DPO, GRPO, KTO, ORPO, SimPO (in that order)
- Columns:
  1. Method name
  2. Paper / year
  3. Requires reference model? (Yes/No)
  4. Requires reward model? (Yes/No)
  5. Data format (paired prefs / unpaired labels / online rollouts)
  6. On-policy or off-policy?
  7. Memory cost (expressed as multiples of single model, e.g., "2×")
  8. Core loss formula (in 1 line of math or pseudocode)
  9. Best use case (1 sentence)
  10. Worst case (1 sentence — when it fails)
- All 6 methods × 10 columns = 60 cells. Every cell must be filled.

**Deliverable:** `week-49-zoo/alignment_comparison.md`

---

## Task 2 — SQL Domain Analysis

**Goal:** Write a 400-word analysis explaining which method to use for each stage of SQL model training.

**Requirements:**
- File: `week-49-zoo/sql_domain_analysis.md`
- Address each of these three questions in 3–5 sentences each:
  1. "Why DPO for the initial preference data (Week 45) and not KTO or SimPO?"
  2. "Why GRPO for the final training stage (Weeks 47–48) and not PPO or DPO?"
  3. "Under what conditions would you switch from GRPO to a different method for your SQL model in the future?"

**Deliverable:** `week-49-zoo/sql_domain_analysis.md`

---

## Task 3 — TRL Trainer Survey

**Goal:** Find the TRL trainer for each of the 6 methods and note the key config parameter that makes each method unique.

**Requirements:**
- For each method, find the TRL trainer class (search docs.huggingface.co/trl)
- Record: trainer class name, key unique config parameter, and its default value
- Note: which trainers are available in TRL vs. which require external implementation

| Method | TRL Trainer | Key Unique Param | Default |
|---|---|---|---|
| PPO | PPOTrainer | kl_coef | 0.05 |
| DPO | DPOTrainer | beta | 0.1 |
| GRPO | GRPOTrainer | num_generations | 8 |
| KTO | ? | ? | ? |
| ORPO | ? | ? | ? |
| SimPO | ? | ? | ? |

**Deliverable:** `week-49-zoo/trl_trainers.md` (filled table)

---

## Stretch Goals

- Implement a 30-step KTO training run on a small model (Qwen-0.5B) using your SQL labels from Week 44. Note: convert your preference pairs to unpaired format (separate `chosen` list as "desirable" and `rejected` list as "undesirable"). How does KTO loss compare to DPO loss on the same data?
- Read the [SimPO experimental results](https://arxiv.org/abs/2405.14734) and find: on the AlpacaEval 2 benchmark, by how many percentage points does SimPO outperform DPO? Does this advantage hold on complex instruction following?
- Browse the TRL changelog (GitHub releases) from 2023 to 2025. List the order in which each trainer was added. What does the order tell you about the adoption trajectory?
