# Week 48 Quiz — Running GRPO

## Multiple Choice

**Q1.** After 300 GRPO steps, you observe that `mean_reward` has increased from 0.18 to 0.24, but `reward_std` has dropped from 0.35 to 0.02. What does this pattern indicate?

A) Training is progressing well — lower variance means the model is more consistent  
B) The model is collapsing — all completions in each group are scoring similarly, producing near-zero gradients  
C) The KL divergence has reached a healthy steady state, reducing reward variance  
D) The learning rate is too high and must be reduced to restore reward variance  

---

**Q2.** Your GRPO training run on RunPod gets disconnected at step 600 of 1000. You resume from the checkpoint at step 550. Which statement best describes the correct procedure?

A) Restart from step 0 — GRPO is sensitive to distribution shift and cannot resume mid-run  
B) Load the step-550 checkpoint, restore optimizer state, and continue training from step 550  
C) Skip directly to step 600 by adjusting the `resume_from_checkpoint` parameter  
D) Resume is not possible in GRPO because the reference model is stateful  

---

**Q3.** After GRPO training, v3 has higher execution accuracy than v2, but its mean generation length increased from 115 to 180 tokens. What is the most likely cause?

A) GRPO optimized for reward-per-token, and longer SQL queries have more chances to include correct patterns  
B) The model learned that longer completions score higher because the reasoning bonus rewards verbosity  
C) The model developed chain-of-thought reasoning before SQL generation, which adds token count  
D) LoRA rank 16 is too high for the 7B model, causing redundant token generation  

---

## Short Answer

**Q4.** KL divergence in your GRPO run climbed from 0.1 to 8.5 nats by step 200, then plateaued. Your eval shows v3 is still better than v2. Should you intervene to reduce KL? What are the risks of leaving it at 8.5 nats?

---

**Q5.** Your GRPO v3 model is 5pp better than v2 on execution accuracy overall, but identical to v2 on the 50-hardest complex queries. The GRPO training prompt set contained only 10% complex queries. Describe what to do in Week 50 to fix this.

---

## Deep Scenario

**Q6.** You have completed your GRPO run with the following results:
- v1 (SFT): 68% execution accuracy, 44% semantic accuracy
- v2 (DPO): 79% execution accuracy, 57% semantic accuracy
- v3 (GRPO): 81% execution accuracy, 55% semantic accuracy

v3 improves on v2 in execution accuracy (+2pp) but is WORSE in semantic accuracy (−2pp). This meets the 5pp threshold with respect to v1 but not relative to v2. GRPO training metrics showed mean_reward grew from 0.22 to 0.38.

Diagnose why v3 has lower semantic accuracy than v2 despite higher execution accuracy. Then propose your Week 50 iteration plan with specific changes to the reward function and training data.
