# Week 59 Answers

## Q1 — Answer: B

**Why:** The DPO loss includes the term `log π_θ(y_r|x) - log π_ref(y_r|x)` inside the sigmoid. When this term is subtracted from the chosen term, minimizing the total loss requires this term to be more negative — meaning `log π_θ(y_r|x)` must become lower than `log π_ref(y_r|x)`. In other words, the model learns to assign lower probability to rejected responses relative to where the reference model puts them. This is the "DPO decreases rejected probability" direction.

---

## Q2 — Answer: B

**Why:** A reward margin of 0.01 at step 200 means the model is barely distinguishing between chosen and rejected. By step 200, you expect the margin to be at least 0.3–0.5 for effective DPO. Root causes: (a) your preference pairs are too similar — both chosen and rejected produce very similar token sequences; (b) the reference model is wrong — you may have the wrong checkpoint loaded as π_ref; (c) the LoRA adapters are being applied incorrectly, so the model is not updating. Check your reference model loading first.

---

## Q3 — Answer: B

**Why:** The execution-correct/wrong-rows pairs are harder negatives: the rejected SQL is plausible enough to execute without error but returns wrong results. The model must learn a more subtle preference signal — not just "avoid syntax errors" but "produce semantically correct SQL." Syntax error pairs are useful (they are real failure cases) but provide an easier learning signal. Weighting the harder pairs 2:1 vs. syntax-error pairs is a reasonable choice.

**Why D is wrong:** Syntax error pairs are perfectly stable for DPO training; they are just lower information density, not harmful.

---

## Q4 — Answer: A (also C is good practice)

**Why:** When you start GRPO training, you need a clean policy checkpoint without the DPO loss accumulated state. Merging the LoRA adapters into the base model weights and then adding fresh LoRA for GRPO is the cleanest approach. Alternatively, you can add new LoRA on top of the DPO model (training a "residual correction") but this compounds adapters and can be unstable.

**Why C is excellent practice:** Running 1,000 generation samples to verify output quality and distribution is a healthy sanity check before any subsequent training stage. If you see distribution collapse (model always generates the same SQL), do not proceed.

---

## Q5 — Model Answer

DPO has overfit to the training pairs. A loss of -2.4 with confidence 0.85/0.02 means the model has completely separated chosen from rejected on training data — but is not generalizing to the eval set. The model has learned to discriminate the specific phrasing patterns of your training pairs, not the underlying SQL correctness.

Actions:
1. Revert to an earlier DPO checkpoint (around step 200, where loss was still near 0) and evaluate it — this is likely a better checkpoint.
2. For the next DPO run: reduce the number of epochs (try 0.5 epochs, ~250 steps), increase beta to 0.5 (stronger KL constraint to prevent collapse), and ensure your training pairs are more diverse (more schemas, more phrasing variation).
3. Inspect 10 training pairs the model is most confident about (loss ≈ -5.0): these are the ones it has "memorized." They are likely phrasing-level near-duplicates. Remove them and re-train.

---

## Q6 — Model Answer

TRL's DPOTrainer implements a "LoRA-as-reference" optimization: when the training model has LoRA adapters and `ref_model=None` is passed, TRL computes the reference model log-probabilities by temporarily disabling the LoRA adapters (setting them to zero contribution) for the reference computation. This works because: `π_ref = base_model` (no LoRA), and `π_θ = base_model + LoRA_adapters`. With LoRA disabled, the base model is the reference.

The constraint: this only works correctly if you have NOT merged the LoRA adapters into the base model weights. If you merged, the "base model with adapters disabled" is no longer equivalent to the original SFT model. Keep the adapters unmerged until after DPO training is complete.

The advantage: you save 14GB of VRAM (no second copy of the 7B model in memory). This is critical on 24GB GPUs.

---

## Q7 — Model Answer

**Strategy A** (teacher chosen, student rejected):
- On-policy: Low. The chosen SQL comes from the teacher (GPT-4o), not the student. The distribution gap between teacher and student can be large for complex TimescaleDB queries.
- Negative hardness: Medium. The rejected SQL is what your model naturally generates — this is real failure cases.
- Best for: Teaching the model to produce teacher-quality SQL style. Most useful early in training when the model is far from teacher quality.
- Risk: Teacher SQL may have style/format that the student can't easily learn to match (off-policy training instability).

**Strategy B** (student correct chosen, student incorrect rejected):
- On-policy: High. Both chosen and rejected come from the student model with temperature > 0.
- Negative hardness: Medium-high. The rejected is what the model naturally generates when it fails.
- Best for: Correcting the model's own failure modes. Most efficient DPO training.
- Risk: If the student rarely generates chosen-level SQL on its own (it needs the teacher's help), this strategy yields few valid pairs.

**Strategy C** (both execute, hard semantic negative):
- On-policy: High (both from student). 
- Negative hardness: Highest. The rejected SQL executes without error, looks reasonable, but returns wrong rows. The model must learn subtle semantic correctness.
- Best for: Pushing from "syntactically correct SQL" to "semantically correct SQL" — the hardest step.
- Risk: Fewer valid pairs (need the student to generate both a correct and an execution-correct-but-wrong answer for the same prompt).

**Recommended strategy:** Use Strategy B as the majority (60% of pairs) — it is the most efficient, on-policy DPO signal. Mix in Strategy C (30%) for the hardest examples to address semantic correctness. Use Strategy A (10%) for TimescaleDB-specific queries where your student still generates poor SQL even with temperature sampling, and the teacher provides the gold standard. This blend maximizes on-policy signal while incorporating teacher guidance where the student is weakest.
