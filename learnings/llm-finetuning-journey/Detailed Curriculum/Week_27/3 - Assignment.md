# Week 27 Assignment — Phase 3 Gate

## Setup Checklist

- [ ] Access to all Phase 3 deliverables: checkpoint, eval.py, dataset, week-24-sota-comparison.md
- [ ] W&B account accessible to retrieve training run URL
- [ ] HuggingFace Hub accessible for dataset URL

---

## Task 1 — Evidence Collection

**Goal:** Gather concrete evidence for each gate criterion.

**Requirements:**

For each criterion, collect the specific evidence item listed below.

**Criterion 1 — 50M LM:**
- Run: `python -c "from model import GPT; m = GPT(); print(sum(p.numel() for p in m.parameters())/1e6, 'M params')`
- Paste the output in `phase3-gate.md`
- Retrieve your W&B run URL and paste it
- Report your best val loss from W&B

**Criterion 2 — Perplexity:**
- Run: `python eval.py --checkpoint checkpoints/best_model.pt --val_path data/val.bin`
- Paste the output showing perplexity number
- Answer from memory (no notes): "What is the formula for perplexity and why do we average loss before exp()?" Write 3 sentences.

**Criterion 3 — Dataset:**
- Run:
```python
from datasets import load_dataset
ds = load_dataset("<your-handle>/postgres-sql-v1")
print(f"Train: {len(ds['train'])}, Val: {len(ds['validation'])}")
```
- Paste the output
- Run your `sql_quality_filter` on the val split and report the pass rate
- Paste 2 example entries from the dataset to show the format

**Criterion 4 — Technical Report Reading:**
- Open `week-24-sota-comparison.md` and verify all 5 model rows are filled
- Without looking at any notes, answer this question in writing: "You are fine-tuning a SQL assistant. Describe the architecture of your chosen base model (Qwen2.5-Coder-7B), its training data characteristics, and 3 reasons why it is better than Llama 3-8B for this task."

**Deliverable:** `Week_27/phase3-gate.md` with all 4 criterion sections filled.

---

## Task 2 — Phase 3 Retrospective

**Goal:** Reflect on what you learned, what was hard, and what you would do differently.

**Requirements:**

Write a 500-word retrospective in `Week_27/phase3-retrospective.md` structured as:

**Section 1: What I built (concrete list)**
- List every artifact produced in Phase 3: files, models, datasets, reports

**Section 2: What I learned that surprised me**
- 2–3 things that were different from what you expected going into Phase 3

**Section 3: What was harder than expected**
- The most technically challenging parts and how you worked through them

**Section 4: What I would do differently**
- If you could redo Phase 3, what 2–3 things would you change?

**Section 5: Gaps I still have**
- Be honest about what you understand conceptually but cannot yet do from memory

**Deliverable:** `phase3-retrospective.md`

---

## Task 3 — Phase 4 Schedule

**Goal:** Plan the next 12 weeks (Phase 4) with realistic time estimates.

**Requirements:**

Write `Week_27/phase4-plan.md` with:
- Week-by-week plan for Weeks 28–40
- For each week: the topic, your main deliverable, and any compute cost estimate
- Identify the weeks where you will need RunPod A100 time (hint: Week 31 QLoRA, Week 35+ full fine-tuning)
- Compute cost estimate for Phase 4 based on the master curriculum

**Deliverable:** `phase4-plan.md`

GitHub commit: `week-27-phase3-gate`

---

## Gate Decision

After completing Tasks 1–3, make your gate decision at the bottom of `phase3-gate.md`:

```
## Final Decision
[ ] READY FOR PHASE 4 — all criteria PASS or acceptable CONDITIONAL
[ ] CONDITIONAL PASS — proceeding with: [list what will be completed in first 2 weeks of Phase 4]
[ ] NEEDS REMEDIATION — will complete [X] before starting Phase 4
    Expected remediation completion: [date]
```

---

## Stretch Goals

- Calculate your actual cumulative compute spend across Phase 3 and compare to the $20 budget in the master curriculum. Did you over/under-spend?
- Write a "lessons learned" post (600 words) on Medium or a private blog about your experience training a 50M language model from scratch
- Preview Phase 4: read the HuggingFace TRL `SFTTrainer` docs and run a 10-step fine-tuning sanity check on `gpt2` with 5 of your dataset examples
