# Week 17 Quiz — Scaling Laws

## Multiple Choice

**Q1.** The Kaplan 2020 scaling laws found that, for a fixed compute budget, you should prioritize:

A) Increasing training tokens while keeping model size small  
B) Increasing model size while training on fewer tokens  
C) Increasing both model size and tokens equally  
D) Increasing learning rate to compensate for smaller models  

---

**Q2.** The Chinchilla paper's key empirical finding was that Gopher (280B params, 300B tokens) was:

A) Over-trained — it needed fewer tokens  
B) Under-trained — it needed ~20× more tokens  
C) Near-optimal — the Kaplan scaling was confirmed  
D) Too small — it needed more parameters  

---

**Q3.** You approximate the FLOP cost of a training run with C ≈ 6ND. Which of the following is NOT captured by this approximation?

A) The forward pass computation  
B) The backward pass computation  
C) Attention's quadratic scaling with sequence length  
D) Parameter update overhead  

---

**Q4.** Llama-3-8B was trained on 15 trillion tokens. Given the Chinchilla 20× heuristic, the compute-optimal token count for an 8B model is roughly 160B tokens. Llama-3-8B is therefore:

A) Under-trained; Meta should have used a larger model  
B) Compute-optimal; 15T ≈ 160B within error  
C) Inference-optimal; Meta traded extra training compute for lower inference cost  
D) Data-poisoned; too many tokens causes catastrophic forgetting  

---

**Q5.** Which quantity does Chinchilla scaling optimize?

A) Inference latency per token  
B) Validation loss per unit of training compute  
C) Model accuracy on MMLU at a fixed parameter count  
D) Training throughput (tokens/sec)  

---

## Short Answer

**Q6.** A team has a compute budget of 1e20 FLOPs. Using Chinchilla Approach 3 constants (a = 6.8e-2, b = 1.96, exponent = 0.5), calculate N_opt and D_opt. Show your work.

---

**Q7.** Explain in 3–4 sentences why a model trained well beyond its Chinchilla-optimal token count (e.g., Llama-3-8B with 15T tokens) is rational from a product perspective, even though it "wastes" training compute.

---

**Q8.** Your manager says: "We have $30 of A100 compute. Let's train a 7B model to keep our options open." Using Chinchilla reasoning, write a 3-sentence rebuttal with numbers.

---

## Scenario

**Q9.** You are advising a startup building a code completion assistant. They have:
- $80 of A100 compute at $1.50/hr
- A high-quality 50B-token code corpus
- A latency requirement of <50ms per completion (so model must be <3B params for their hardware)

Using Chinchilla analysis:
1. What is the compute budget in FLOPs? (assume 35% MFU)
2. What does Chinchilla say is the optimal N? Is it feasible given the <3B constraint?
3. Given the 50B-token data constraint, what is the maximum useful model size?
4. What would you recommend they actually train, and why?
