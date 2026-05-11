# Week 46 Quiz — GRPO and RLVR

## Multiple Choice

**Q1.** In GRPO, the advantage for completion y_i in a group of K completions is computed as:

A) A_i = r_i − V(s_i), where V is a learned critic network  
B) A_i = (r_i − mean(r_1,...,r_K)) / std(r_1,...,r_K)  
C) A_i = r_i − r_ref, where r_ref is the reward of the reference model's most likely completion  
D) A_i = r_i / Z, where Z is the partition function normalizing all rewards in the batch  

---

**Q2.** GRPO generates K=8 completions for a given SQL prompt. All 8 execute correctly (reward = 1 for all). What is the gradient update for this step?

A) The model receives a large positive gradient, reinforcing all 8 completions equally  
B) The gradient is zero — when all rewards are equal, std = 0 and all advantages are 0  
C) The gradient is negative — the model is penalized for generating duplicate completions  
D) The gradient is undefined because division by zero occurs in the normalization  

---

**Q3.** Why did GRPO, not DPO, become the preferred method for training DeepSeek-R1?

A) DPO cannot be applied to large models; GRPO scales better with model size  
B) DPO requires human preference data; for math problems with ground-truth answers, GRPO uses verifiable rewards without any labeling  
C) GRPO uses less memory than DPO because it does not require a reference model  
D) DPO's β hyperparameter is unstable for large datasets; GRPO has no equivalent instability  

---

**Q4.** The DeepSeek-R1 paper found that GRPO training on math problems caused the model to spontaneously develop extended reasoning chains. What is the mechanism?

A) GRPO explicitly trains the model to generate chain-of-thought by rewarding intermediate reasoning steps  
B) The verifiable reward only evaluates the final answer; the model discovers that reasoning chains improve final answer accuracy, and the correct answers get higher rewards  
C) DeepSeek-R1 uses a special "reasoning reward" that scores the quality of intermediate reasoning text  
D) GRPO's group sampling encourages diverse reasoning paths, which are filtered to select the best reasoning chain  

---

**Q5.** GRPO uses a KL penalty to the reference model (same as PPO-RLHF). If you remove this KL penalty entirely, what is the most likely failure mode?

A) The model cannot learn because the gradient becomes zero without the KL term  
B) The model drifts so far from the reference distribution that it generates incoherent text or exploits the reward function  
C) The group-relative normalization becomes unstable because rewards are no longer bounded  
D) The model converges to a deterministic policy that always generates the same completion  

---

## Short Answer

**Q6.** Explain in concrete terms why GRPO does not need a critic network for SQL training, while PPO does. Reference the verifiable reward property specifically.

---

**Q7.** For SQL generation with GRPO and K=8, you observe that for 70% of prompts, all 8 completions either all execute (reward=1) or all fail (reward=0). This means 70% of training steps have zero gradient. Propose two interventions to increase training signal.

---

## Deep Scenario

**Q8.** You are designing a GRPO training run for SQL generation. A colleague proposes using K=2 (generate only 2 completions per prompt) to save compute. You propose K=16. Argue your position:
1. Explain the theoretical statistical advantage of K=16 over K=2
2. Give the worst-case scenario for K=2 that does not apply to K=16
3. Estimate the compute cost ratio (K=16 vs K=2) and justify why it is worth it for your SQL use case
