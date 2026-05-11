# Week 45 Quiz — DPO on Your Domain Model

## Multiple Choice

**Q1.** After DPO training on your SQL preference dataset, `rewards/chosen` is increasing but `rewards/rejected` is also increasing (becoming less negative). The most actionable intervention is:

A) Increase β to keep the model closer to the reference  
B) Decrease the learning rate by 10×  
C) Verify that the reference model weights are truly frozen and not receiving gradient updates  
D) Switch from Unsloth DPO to TRL vanilla DPO for better numerical stability  

---

**Q2.** Your DPO model (v2) shows improved execution accuracy on simple queries (+12pp) but identical performance on complex queries. Which statement best explains this outcome?

A) The LoRA rank was too low to capture complex SQL patterns  
B) The preference dataset contains mostly simple query pairs; DPO has no signal for complex patterns not in the dataset  
C) The β value was too high, constraining the model from learning complex SQL syntax  
D) Complex queries require a larger model; 7B parameters is insufficient  

---

**Q3.** DPO loss goes negative at step 150 of your SQL training run. The most likely cause is:

A) The reward_margin exceeded 1.0, causing numerical overflow in the log-sigmoid  
B) Some preference pairs are mislabeled — rejected SQL has higher log-probability than chosen SQL  
C) The learning rate is too low, causing the optimizer to overshoot the loss minimum  
D) The DPO loss is inherently negative when β < 0.1 — this is expected behavior  

---

## Short Answer

**Q4.** You are comparing v1 (SFT) and v2 (DPO) on your test set. v2 has higher execution accuracy but lower semantic accuracy. Explain how this is possible and what it implies about your preference dataset quality.

---

**Q5.** Your v2 model starts generating refusals ("I cannot generate SQL for this request") on about 10% of test prompts, while v1 never refused. You did not include any refusal data in your preference dataset. Propose a diagnosis and a concrete fix.

---

**Q6.** Describe how you would use the eval results from this week to guide the GRPO reward function design in Weeks 47–48. Specifically: what failure modes of v2-dpo does GRPO need to fix, and how does each failure mode translate to a reward function design choice?

---

## Deep Scenario

**Q7.** You have trained v2-dpo and the eval shows:
- Execution accuracy: v2 = 71%, v1 = 72% (v2 is WORSE by 1pp)
- Semantic accuracy: v2 = 48%, v1 = 53% (v2 is WORSE by 5pp)
- reward_margin during training: reached 0.12 then plateaued
- Training loss: oscillated between 0.60 and 0.65 without clear trend

This is a DPO failure. Work through a systematic diagnosis:
1. What does the low reward_margin plateau indicate?
2. What does oscillating loss without trend indicate?
3. Propose 3 interventions, ranked by likelihood of fixing the problem, with specific hyperparameter changes for each.
