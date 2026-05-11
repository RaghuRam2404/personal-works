# Week 49 Quiz — Alignment Zoo

## Multiple Choice

**Q1.** You have 50,000 SQL examples labeled as "good" from expert engineers and 20,000 labeled as "bad" from error logs, but they are not paired by prompt. Which alignment method is best suited to this data format?

A) DPO — convert to paired format by randomly matching good and bad examples  
B) GRPO — use the "good" examples as the reward signal  
C) KTO — designed for unpaired good/bad annotation datasets  
D) ORPO — the monolithic loss handles unpaired data natively  

---

**Q2.** ORPO eliminates the reference model. What is the primary downside of this?

A) ORPO cannot be used with LoRA adapters  
B) Without a reference model, there is no KL regularization anchor, so the model can drift far from the pretrained distribution  
C) ORPO requires 3× more memory than DPO because it must maintain the SFT and preference losses separately  
D) ORPO converges slower than DPO because the odds ratio is harder to optimize  

---

**Q3.** SimPO normalizes the log-probability reward by sequence length: reward = (1/|y|) · log π_θ(y|x). What problem does this solve that standard DPO does not?

A) SimPO prevents reward hacking from information_schema queries  
B) SimPO makes the reward independent of sequence length, preventing the model from systematically preferring shorter or longer completions  
C) SimPO eliminates the need for the reference model  
D) SimPO handles unpaired annotations better than DPO  

---

**Q4.** A colleague proposes switching from GRPO to DPO for your SQL training because "DPO is simpler and cheaper." The best argument against this switch is:

A) DPO requires more memory than GRPO because it uses two forward passes  
B) DPO is an offline method that cannot adapt to SQL patterns discovered during training; GRPO's online rollouts provide fresh execution-based rewards at every step  
C) DPO loss can go negative, which GRPO's loss cannot  
D) DPO requires human preference labelers, which GRPO does not  

---

## Short Answer

**Q5.** A new alignment method paper claims: "Our method requires no reference model, no reward model, and works with unpaired data." Based on this week's survey, describe what tradeoffs this method is likely making (even without reading the paper). Reference KTO, ORPO, and SimPO in your answer.

---

**Q6.** Compare the data efficiency of DPO vs. KTO. If you have 1000 paired preference examples, how many data points does DPO see per step? How many does KTO see? Which is more efficient per data point?

---

## Deep Scenario

**Q7.** You are working at a startup that has just deployed your v3 SQL model in production. After 3 months, you have:
- 100,000 SQL queries that were executed successfully by users (positively labeled)
- 30,000 SQL queries that were manually corrected by users before execution (negatively labeled)
- The positively labeled and negatively labeled queries are for DIFFERENT prompts (not paired)

You want to train v4 using this production data. Walk through your decision process: which alignment method would you use, and why? Include a discussion of at least 3 methods you considered and rejected.
