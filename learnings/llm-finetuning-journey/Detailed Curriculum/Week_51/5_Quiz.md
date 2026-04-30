# Week 51 Quiz — Final Iteration and Model Selection

## Multiple Choice

**Q1.** You have run 5 experiments in Weeks 50–51. The last 3 experiments each improved execution accuracy by 0.5pp or less. According to the stopping rules, you should:

A) Run a 6th experiment targeting a different metric (semantic accuracy) since execution accuracy is plateauing  
B) Stop iterating — diminishing returns indicate you have reached the current ceiling; pick the best checkpoint  
C) Double the number of GRPO steps in the next experiment to overcome the plateau  
D) Switch from GRPO to ORPO for the next experiment to get a different optimization signal  

---

**Q2.** Your final eval shows v3-iter2 has 85% execution accuracy and 58% semantic accuracy. v3-iter1 has 84% execution accuracy and 61% semantic accuracy. Which is the better model for the Phase 5 Gate, and what is the deciding factor?

A) v3-iter2 — execution accuracy is the primary Gate criterion  
B) v3-iter1 — semantic accuracy reflects actual correctness; a model that executes but returns wrong rows is less useful  
C) v3-iter2 — 1pp execution accuracy difference is more significant than 3pp semantic accuracy difference  
D) v3-iter1 — the Gate requires semantic accuracy above 60%, so v3-iter1 is the only one that passes  

---

**Q3.** When producing your final eval report, you discover that you ran experiments targeting the same 200-query eval set across 4 iterations. A colleague says your results may overfit to the eval set. What is the appropriate response?

A) Discard all results and start with a fresh 200-query eval set  
B) Acknowledge the limitation in the report and test the best model on 50 additional held-out queries you did not use during iteration  
C) Run a 5th experiment to "validate" the results  
D) The limitation is not real — an eval set cannot be overfit if you use greedy decoding  

---

## Short Answer

**Q4.** Your Phase 5 best model (v3-iter2) has higher KL divergence (5.1 nats) than v3 (3.2 nats). The eval metrics are better on v3-iter2. Write 2–3 sentences explaining what the higher KL implies for Phase 6 (when you will restart training with a larger dataset), and whether it is a concern.

---

**Q5.** You are comparing your best Phase 5 SQL model against GPT-4o on 20 queries. On single-table queries, your model wins 12/20 (60%). On 3+ JOIN queries, your model wins 4/20 (20%). What does this comparison tell you about the priority for Phase 6 dataset construction?

---

## Deep Scenario

**Q6.** You are writing the Phase 5 retrospective for your GitHub/portfolio. You need to tell a technically honest story about the journey from v1 to v3. Write 4–5 sentences summarizing:
1. What each alignment stage (SFT, DPO, GRPO) contributed
2. The most important lesson learned about reward function design
3. One thing that did not work as expected and why
4. What you would do differently if starting Phase 5 over with the knowledge you have now
