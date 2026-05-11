# Week 60 Quiz — GRPO Final

## Multiple Choice

**Q1.** GRPO with group size K=8 generates 8 SQL candidates per prompt. The rewards are: [2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1] — all identical. What is the GRPO gradient update for this step?

A) A strong positive update — all 8 candidates received high reward.
B) Zero — GRPO uses reward differences within the group; if all rewards are equal, the advantage is zero.
C) A negative update — the model is overconfident and GRPO penalizes this.
D) A normal update proportional to the mean reward of 2.1.

---

**Q2.** You choose to run GRPO on 1,500 prompts where your DPO-v3 model currently fails. A colleague suggests using ALL 25,000 v3 training examples as GRPO prompts. What is the main practical problem with the colleague's suggestion?

A) GRPO training on > 5,000 examples always causes catastrophic forgetting.
B) The per-prompt cost of GRPO is K× higher than SFT; 25,000 × K=8 examples would take impractically long on H100.
C) GRPO requires unique prompts; the v3 dataset has too many near-duplicates.
D) The reward function cannot handle more than 2,000 Postgres connections simultaneously.

---

**Q3.** After 200 GRPO steps, your KL divergence from the reference (DPO-v3) is 12.4 bits. The policy has drifted significantly. What should you do?

A) Continue training — KL divergence always increases and the model is learning.
B) Stop training, increase kl_coef from 0.05 to 0.15, and resume from the step-100 checkpoint.
C) Reduce the learning rate to 1e-7 to slow the KL divergence increase.
D) KL divergence > 10 is expected and safe for GRPO; continue training.

---

**Q4.** Your GRPO reward function gives partial credit (0.2) when the generated SQL executes but returns the wrong number of rows. Explain why partial credit is better than a binary reward for GRPO training with small group size (K=4).

---

## Short Answer

**Q5.** Your final GRPO model achieves 78% execution accuracy on your custom benchmark. GPT-4o achieves 83% on the same benchmark. Is your project a success? Justify your answer — consider what "success" means for a 7B domain-specialist vs. a 200B generalist.

---

**Q6.** You merge the GRPO LoRA adapters into the base model using `save_method="merged_16bit"`. The merged model file is 14.2GB. A colleague's model with the same architecture is 28.4GB. What is the difference and is either incorrect?

---

## Deep Scenario

**Q7.** You have trained four models: SFT-v3, DPO-v3, GRPO-final, and (from Phase 5) Phase5-GRPO. You are writing your technical report. Your evaluation results are:

| Model | Custom 200 | TimescaleDB subset | BIRD-SQL dev |
|-------|-----------|-------------------|--------------|
| Phase5-GRPO | 59% | 41% | 52% |
| SFT-v3 | 71% | 63% | 67% |
| DPO-v3 | 74% | 67% | 70% |
| GRPO-final | 76% | 70% | 71% |

Write a 3-paragraph evaluation discussion for your technical report. Address: (a) what each training stage contributed, (b) why the TimescaleDB subset numbers are lower than the overall custom benchmark, and (c) what the remaining gap to GPT-4o (83%) likely represents.
