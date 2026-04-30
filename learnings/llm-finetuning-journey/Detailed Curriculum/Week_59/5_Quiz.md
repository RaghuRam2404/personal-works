# Week 59 Quiz — DPO

## Multiple Choice

**Q1.** The DPO loss formula contains a term `log π_θ(y_r|x)/π_ref(y_r|x)`. What is the effect of minimizing this term during training?

A) It encourages the model to assign higher probability to rejected responses.
B) It encourages the model to assign lower probability to rejected responses relative to the reference model.
C) It forces the model's distribution to match the reference model exactly.
D) It has no effect — only the chosen term contributes to learning.

---

**Q2.** After 200 DPO steps, your W&B shows: `rewards/chosen = 0.12`, `rewards/rejected = 0.11`. The reward margin is 0.01. What does this tell you?

A) DPO is working correctly — small margins are expected at 200 steps.
B) The model is barely distinguishing chosen from rejected; the pairs are too similar or the reference model is misconfigured.
C) Beta is too high — reduce it to allow larger reward margins.
D) The training is converging too fast; add more regularization.

---

**Q3.** You have 2,000 preference pairs where both chosen and rejected execute correctly (chosen has higher result accuracy, rejected has wrong rows). You have 2,000 pairs where chosen executes and rejected has a syntax error. Which type should you weight more heavily in DPO training and why?

A) Equal weighting — both types provide useful signal.
B) Weight the execution-correct/wrong-rows pairs more heavily — these are harder, more informative negatives.
C) Weight the syntax-error pairs more heavily — preventing syntax errors is more important for production.
D) Use only the execution-correct/wrong-rows pairs — syntax error pairs cause DPO instability.

---

**Q4.** You want to use the DPO-trained model as the starting point for GRPO in Week 60. What must you do before starting GRPO?

A) Merge the DPO LoRA adapters into the base model to create a clean starting checkpoint.
B) Reset the model to the SFT-v3 checkpoint — GRPO always starts from SFT, not DPO.
C) Run the DPO model on 1,000 generation samples to verify no distribution collapse before GRPO.
D) Nothing special — GRPO can start from any model checkpoint without modification.

---

## Short Answer

**Q5.** Your DPO loss is `-2.4` at step 500. The model assigns probability 0.85 to chosen and 0.02 to rejected on the training pairs. Execution accuracy on held-out eval is 70% — exactly the same as before DPO. What is happening and what do you do?

---

**Q6.** DPO requires a reference model `π_ref`. With LoRA, you can avoid storing a separate copy of the reference model. How does TRL's DPOTrainer implement this optimization with LoRA, and what is the constraint it imposes?

---

## Deep Scenario

**Q7.** You are comparing three preference labeling strategies for your SQL domain:

Strategy A: Chosen = teacher-generated SQL; Rejected = student (SFT-v3) incorrect output
Strategy B: Chosen = student correct output; Rejected = student incorrect output (both on-policy)
Strategy C: Chosen = student output that executes correctly; Rejected = student output that executes but returns wrong rows (both execute, hard negative)

For each strategy, analyze: (a) how on-policy the data is, (b) how hard the negatives are, (c) which failure modes it best addresses, and (d) which strategy you would use for your production DPO run and why.
