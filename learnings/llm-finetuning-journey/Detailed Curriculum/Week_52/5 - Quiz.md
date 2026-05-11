# Week 52 Quiz — Phase 5 Gate Assessment

This is the gate quiz. All questions must be answered. Minimum passing standard: all MCQ correct AND all short-answer questions answered with mathematical precision.

## Multiple Choice (Senior Applied Level)

**Q1.** Complete the derivation: in DPO, the reparameterized reward is r(x,y) = β·log(π*(y|x)/π_ref(y|x)) + β·log Z(x). When computing r(x, y_w) − r(x, y_l) in the Bradley-Terry loss, Z(x) disappears because:

A) Z(x) = 1 by normalization of the reference model  
B) Z(x) is the same constant for both y_w and y_l (it depends only on x), so it cancels in the difference  
C) Z(x) is approximated as exp(0) = 1 in the DPO derivation  
D) Z(x) cancels with the normalization constant in the Bradley-Terry sigmoid  

---

**Q2.** Your DPO model (v2) has higher accuracy on chosen responses (the chosen SQL executes correctly more often) but generates more refusals ("I cannot answer this") on 10% of test prompts. v1 (SFT) never refuses. The most likely cause and fix:

A) DPO's β is too high — lower it to allow more divergence from π_ref, which will reduce conservative behavior  
B) The rejected examples in the preference dataset contain refusal patterns; DPO reduced their probability along with the valid rejected responses  
C) The SFT model was trained on instruction data that included refusals; DPO amplified this bias  
D) The reference model (frozen v1) generates refusals that the KL penalty penalizes the training model for not generating  

---

**Q3.** Compare PPO vs. DPO vs. GRPO for a SQL text-to-SQL system with: (1) real Postgres execution available, (2) no human labelers, (3) 32GB VRAM budget. Which is best and why?

A) PPO — most established algorithm with the most tooling  
B) DPO — lowest memory (2×) and no online generation required  
C) GRPO — verifiable reward (SQL execution) + no reward model + no critic = optimal for this setting  
D) KTO — works with unpaired labels which SQL execution naturally produces  

---

**Q4.** GRPO generates K=8 completions for the prompt "Show revenue by quarter for 2024." All 8 completions execute correctly (reward = 1.0). The training gradient is:

A) Positive and large — all correct completions should be strongly reinforced  
B) Zero — with all rewards equal, all advantages are 0 and the policy gradient is 0  
C) Negative — GRPO penalizes mode collapse when all completions are identical  
D) Positive but small — the mean reward being 1.0 gives a slight positive advantage  

---

**Q5.** Your GRPO model's `reward_std` collapsed to near-zero at step 400. The first intervention to try is:

A) Increase the KL coefficient β from 0.05 to 0.5  
B) Switch to DPO for the remaining training  
C) Increase generation temperature from 0.7 to 0.9 to force more diverse completions  
D) Reduce K from 8 to 4 to reduce the number of homogeneous completions  

---

## Short Answer (Mathematical Precision Required)

**Q6.** Write the GRPO advantage formula. Show that when rewards = [1, 0, 1, 0], the advantages are ≈ [0.866, −0.866, 0.866, −0.866] (using sample std with ddof=1). Show your arithmetic.

---

**Q7.** Explain in 4 sentences why GRPO does not need a critic network for SQL training, while PPO does. Reference: (1) what the critic does in PPO, (2) what replaces it in GRPO, (3) why the replacement is valid for SQL specifically, (4) when the replacement would NOT be valid.

---

**Q8.** Design a reward function for SQL with the following properties:
1. Binary execution success is not sufficient — the query must return rows matching the expected output
2. Partial credit for getting the right row count even if values differ
3. Anti-hack guard preventing information_schema queries from scoring above 0
4. The function must run in < 500ms per query
5. A reasoning chain before the SQL earns a small bonus, but not more than 10% of max reward

Write the reward levels (0.0, 0.1, 0.2, 0.5, 1.0, and optional bonus) with their conditions. Your answer should be concrete enough to implement directly.

---

## Deep Scenario

**Q9.** You have completed Phase 5. Your best model (v3-iter2) achieves 84% execution accuracy, 60% semantic accuracy, and 62% complex query accuracy — all significantly better than v1 (68%, 44%, 40%). You are now planning Phase 6.

A colleague argues: "You should skip Phase 6's DPO stage and go straight from SFT to GRPO, since DPO didn't help complex queries anyway."

Evaluate this argument. Under what conditions is the colleague right? Under what conditions are they wrong? What would you need to measure to decide?
