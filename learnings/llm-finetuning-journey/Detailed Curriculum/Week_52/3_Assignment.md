# Week 52 Assignment — Phase 5 Gate

This is the gate. Complete every task. Do not advance to Phase 6 until all are done.

---

## Gate Task 1 — Written Mathematical Derivations

**Goal:** Prove you understand the math, not just the recipes.

**Requirements:**
- File: `week-52-gate/mathematical_derivations.md`
- Complete all three derivations in writing (typed in Markdown with LaTeX math, or scanned handwritten):

**Derivation 1: DPO loss from KL-constrained RL objective**
- Show all 6 steps from the curriculum (KL objective → optimal policy → reward reparameterization → Z(x) cancels → preference loss → DPO loss)
- No shortcuts. Show every algebraic step.

**Derivation 2: REINFORCE policy gradient theorem**
- Start from J(θ) = E[R(τ)]
- Apply the log-derivative trick
- Arrive at the REINFORCE gradient estimator
- Explain why this is valid even when R is non-differentiable (e.g., SQL execution)

**Derivation 3: GRPO advantage formula and its connection to REINFORCE baseline subtraction**
- State the GRPO advantage: A_i = (r_i - mean(r)) / std(r)
- Show it is a specific instance of REINFORCE baseline subtraction
- Explain: why does the advantage become 0 when all K rewards are equal?
- Explain: why does a deterministic verifiable reward make the within-group mean a good baseline?

**Acceptance criterion:** All three derivations are complete and mathematically correct. No missing steps.

---

## Gate Task 2 — Model Verification

**Goal:** Verify all three pipeline models exist and are accessible.

**Requirements:**
- Log into HuggingFace Hub and verify:
  - `<your-handle>/postgres-sqlcoder-7b-v1` — SFT model
  - `<your-handle>/postgres-sqlcoder-7b-v2-dpo` — SFT + DPO
  - `<your-handle>/postgres-sqlcoder-7b-phase5-best` — Best Phase 5 model (GRPO)
- For each model, run:
  ```python
  from transformers import AutoModelForCausalLM, AutoTokenizer
  tok = AutoTokenizer.from_pretrained("<model_path>")
  m = AutoModelForCausalLM.from_pretrained("<model_path>")
  out = m.generate(tok("SELECT all users:", return_tensors="pt").input_ids, max_new_tokens=50)
  print(tok.decode(out[0]))
  ```
- Verify all three models load and generate output.
- Verify the Phase 5 final eval report exists: `week-51-iteration/phase5_final_report.md`
- Verify the report shows: `phase5-best` execution accuracy > v1 execution accuracy

**Deliverable:** `week-52-gate/model_verification.md` — output from the generation test for each model, plus a statement confirming the eval report pass condition.

---

## Gate Task 3 — GRPO Run Verification

**Goal:** Confirm you have run GRPO with executable rewards.

**Requirements:**
- Provide a link to your Week 48 W&B run (`week-48-grpo-sql` project)
- The run must show: at least 500 steps logged, `mean_reward` metric present, `reward_std` metric present
- If W&B is not accessible, paste the first 10 lines of the training log from Week 48

**Deliverable:** `week-52-gate/grpo_verification.md` — W&B link or training log excerpt

---

## Gate Task 4 — Phase 5 Reflection

**Goal:** Write a structured reflection on Phase 5 as a whole.

**Requirements:**
- File: `week-52-gate/phase5_reflection.md`
- Answer each question in 3–5 sentences:
  1. What was the most important thing you learned in Phase 5 that you did not know at the end of Phase 4?
  2. Which week was the hardest? Why? What did you do to get through it?
  3. What was the biggest surprise (something that did not work as you expected)?
  4. What is the one thing you would do differently in Phase 5 if you started over?
  5. What is the specific question you most want to answer in Phase 6?

---

## Gate Task 5 — Phase 6 Preview

**Goal:** Preview Phase 6 to ensure you know where you are going.

**Requirements:**
- Read the Phase 6 section of the master curriculum (`llm_finetuning_curriculum_18months.md`)
- Write `week-52-gate/phase6_preview.md` with:
  - The Phase 6 goal (1 sentence)
  - The three biggest compute investments in Phase 6 (with cost estimates)
  - What "beats GPT-4 on your domain" means concretely — how will you measure it?
  - One technical risk you see in Phase 6 and how you plan to mitigate it

**Deliverable:** `week-52-gate/phase6_preview.md`

---

## Gate Pass Criteria (all must be met)

- [ ] Task 1: All three derivations complete and correct
- [ ] Task 2: All three models verified, eval report confirms v3 > v1
- [ ] Task 3: GRPO run with executable rewards confirmed
- [ ] Task 4: Reflection complete (5 questions answered)
- [ ] Task 5: Phase 6 preview complete

If all boxes are checked: **You have passed Phase 5. Advance to Phase 6 (Week 53).**

If any box is unchecked: Return to the relevant week(s). Repeat if needed.
