# Week 41 Quiz — RL Primer

## Multiple Choice

**Q1.** In the RLHF pipeline for a language model, what is the "action" at each time step?

A) The entire completion generated in response to a prompt  
B) A single token selected from the vocabulary  
C) The scalar score output by the reward model  
D) A gradient update applied to the model weights  

---

**Q2.** The log-derivative trick transforms ∇_θ p(τ|θ) into:

A) p(τ|θ) · R(τ)  
B) p(τ|θ) · ∇_θ log p(τ|θ)  
C) ∇_θ log p(τ|θ) · ∇_θ R(τ)  
D) R(τ) · ∇_θ p(τ|θ) / p(τ|θ)  

---

**Q3.** You train a REINFORCE agent on a task where reward is only given at the very last step of a 200-step episode (sparse reward). Which of the following is the most direct consequence?

A) The model cannot learn at all because REINFORCE requires dense rewards  
B) The gradient variance is extremely high because all 200 steps get the same return G_0  
C) The discount factor γ must be set to exactly 0 for the agent to converge  
D) The Markov property is violated, so the MDP formulation breaks down  

---

**Q4.** Which statement correctly distinguishes on-policy from off-policy learning?

A) On-policy methods use a replay buffer; off-policy methods do not  
B) Off-policy methods can only be used with discrete action spaces  
C) On-policy methods update using data sampled from the current policy; off-policy methods can use data from older policies  
D) On-policy methods require a critic network; off-policy methods do not  

---

**Q5.** Return normalization in REINFORCE (subtracting mean and dividing by std) primarily helps by:

A) Ensuring the discount factor γ is always between 0 and 1  
B) Converting the return into an unbiased estimator of the advantage function  
C) Reducing gradient variance without introducing bias into the gradient estimate  
D) Guaranteeing the agent converges in fewer than 500 episodes  

---

## Short Answer

**Q6.** Write the REINFORCE loss in PyTorch pseudocode for one episode. Your answer should show: how log probabilities are collected, how the return is computed, and what the loss expression is.

---

**Q7.** The advantage function is defined as A(s, a) = Q(s, a) − V(s). In the REINFORCE context, what acts as the advantage, and what is the practical effect of using it instead of the raw return G_t?

---

**Q8.** A colleague says: "We do not need RL for RLHF because DPO does not involve any RL training loop." Evaluate this statement. What does DPO actually optimize, and why does RL theory still matter for understanding it?

---

## Scenario

**Q9.** You are applying REINFORCE to fine-tune a 1B-parameter LLM to generate executable SQL. Your reward function returns +1 if the SQL executes correctly on a test Postgres database, 0 otherwise. After 500 episodes, the model's average reward is stuck at 0.08 (8% of generations execute correctly) and is not improving.

Propose three specific interventions, ranked by likelihood of impact. For each, explain the RL-theoretic reason why it should help.
