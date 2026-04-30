# Week 42 Quiz — PPO and RLHF

## Multiple Choice

**Q1.** In the PPO clipping objective, the ratio r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t) is clipped to [1−ε, 1+ε]. When the advantage A_t is positive, what does the clip do?

A) It prevents the new policy from assigning lower probability to the action than the old policy  
B) It prevents the new policy from assigning much higher probability to the action than the old policy  
C) It prevents the reward from exceeding the KL budget  
D) It has no effect when A_t is positive — clipping only applies to negative advantages  

---

**Q2.** In InstructGPT's RLHF, the KL penalty β · KL(π_θ || π_SFT) is added to the reward. What happens if β is set too close to zero?

A) The model forgets how to generate coherent text because there is no regularization  
B) The model converges faster because the penalty gradient does not interfere with reward maximization  
C) The model cannot learn because the value function becomes undefined  
D) The model collapses to a single token output because the vocabulary softmax concentrates  

---

**Q3.** GAE with λ=0 reduces to which estimator?

A) Full Monte Carlo return (high variance, low bias)  
B) 1-step TD residual (low variance, high bias)  
C) Importance-weighted return from the old policy  
D) The raw reward at the current step with no value baseline  

---

**Q4.** The reference model (π_ref) in TRL's PPOTrainer is:

A) Updated every epoch using a slow-moving average of the training model  
B) A frozen copy of the SFT model used only to compute KL divergence  
C) Replaced by the reward model once training is stable  
D) The same object as the value (critic) model  

---

## Short Answer

**Q5.** Write the complete PPO reward for a language model completion at position t in terms of: the reward model score r_RM, the KL divergence KL_t at position t, and the penalty coefficient β. Explain where the reward model score is applied (at every token or only at the final token) and where the KL penalty is applied.

---

**Q6.** Explain why training an LLM with PPO requires four simultaneous forward passes (actor, critic, reward model, reference model). For each, state what it computes and why you cannot eliminate it.

---

**Q7.** Your PPO RLHF run on a SQL generation model shows: reward model score is climbing, but when you manually evaluate 50 generated queries, the SQL quality is getting worse (more hallucinated table names, more syntax errors). Name this phenomenon and propose two interventions.

---

## Deep Scenario

**Q8.** You are training a 7B SQL generation model with PPO+RLHF. After 1000 PPO steps, you observe:
- KL divergence from the reference model has grown to 45 nats (extremely high)
- Reward model score is at 8.2 / 10 (suspiciously high)
- Manually, the model now generates SQL like: `SELECT * FROM users WHERE 1=1; -- reward hack pattern`
- The value loss is oscillating wildly

Diagnose what went wrong at each component level (KL, reward model, value function) and provide a concrete remediation plan with specific hyperparameter adjustments.
