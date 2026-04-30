# Week 43 Quiz — DPO: Direct Preference Optimization

## Multiple Choice

**Q1.** In the DPO derivation, the partition function Z(x) appears in the closed-form optimal policy. Why does it not appear in the final DPO loss?

A) Z(x) is approximated as 1.0 for computational tractability  
B) Z(x) cancels when computing the difference r(y_w) − r(y_l) because Z depends only on x, not y  
C) Z(x) is absorbed into the β hyperparameter during reparameterization  
D) Z(x) is replaced by the reference model's normalization constant  

---

**Q2.** During DPO training, you observe that `rewards/chosen` is increasing but `rewards/rejected` is also increasing (less negative than expected). What does this indicate?

A) The model is performing well — both log-ratios increasing means better calibration  
B) The reference model weights are being accidentally updated  
C) β is too high, over-regularizing the model toward the reference distribution  
D) The model is collapsing toward the reference model and not learning the preference signal  

---

**Q3.** DPO uses a lower learning rate (5e-7) than SFT (1e-5 to 2e-4). What is the primary reason?

A) DPO loss gradients are larger in magnitude than cross-entropy gradients  
B) The model must stay close to π_ref; large updates would defeat the KL constraint built into the loss  
C) DPO is only used for LoRA training, which requires lower learning rates  
D) The ultrafeedback dataset is noisy and a lower LR prevents overfitting  

---

**Q4.** You train a DPO model and find it has higher accuracy on chosen responses but generates more refusals ("I cannot do that"). What is the most likely cause?

A) β is too low, causing the model to overfit to rejected response patterns  
B) The reference model was initialized from a different checkpoint than the SFT model  
C) The "rejected" responses in the preference dataset often contain refusals, and DPO is reducing their probability along with the valid rejected responses  
D) DPO is not compatible with instruction-tuned base models  

---

**Q5.** Which of the following tasks is DPO LEAST well-suited for compared to PPO or GRPO?

A) Aligning a chat model to be more helpful based on human preference data  
B) Teaching a model to generate code that passes automated unit tests  
C) Making a model less toxic based on pairwise human annotations  
D) Adjusting the style of a model's outputs to match user preferences  

---

## Short Answer

**Q6.** Write the DPO loss formula in compact notation. Then identify the two components that TRL's DPOTrainer computes in `compute_loss()` — what are `logps_chosen` and `logps_rejected`, and which model(s) compute them?

---

**Q7.** DPO assumes the Bradley-Terry preference model. State the two key assumptions of Bradley-Terry and explain one way each could be violated in a real SQL preference dataset.

---

**Q8.** Compare DPO and PPO on three dimensions: (1) data requirements, (2) compute per training step, (3) ability to handle verifiable rewards. For each dimension, state which is better and why.

---

## Deep Scenario

**Q9.** You have trained a DPO model on your SQL preference dataset. Evaluation shows:
- On "easy" queries (single table, no joins), v2-dpo is 15pp better than v1-sft
- On "hard" queries (3+ table joins, subqueries), v2-dpo performs identically to v1-sft or worse
- The reward margin during DPO training plateaued at 0.3 (low)

Diagnose why DPO is failing on hard queries and propose two interventions. One intervention must involve the preference dataset design; the other must involve the DPO training configuration.
