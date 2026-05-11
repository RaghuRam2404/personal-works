# Week 68 Quiz — Training Pipeline Documentation

## Multiple Choice

**Q1.** A reviewer comments: "The paper does not report gradient clipping values or optimizer weight decay, making the training unstable and irreproducible." This feedback is directed at which section of your report?

A. The dataset section — gradient clipping is a data preprocessing concern
B. The evaluation section — these values affect test-time behavior
C. The training pipeline section — specifically the hyperparameter table
D. The limitations section — it is acceptable to omit optimizer details

**Q2.** You report "training ran for 2,400 steps on 2× A100-40GB GPUs." A reproducibility reviewer wants to know the approximate wall-clock time. Which additional piece of information, combined with your step count, enables them to compute it?

A. The final validation loss
B. The number of tokens processed per second (or per step)
C. The LoRA rank
D. The learning rate schedule

**Q3.** Your DPO section says "β=0.1." Another researcher is confused because their DPO implementation uses a parameter called `reference_free` and does not expose a beta. What should your section clarify?

A. That your DPO is reference-free and β is irrelevant
B. That β=0.1 controls the KL penalty between the trained policy and the reference (frozen) policy in standard DPO
C. That β=0.1 is equivalent to a learning rate multiplier
D. That β should be tuned between 0.01 and 1.0 for each task

**Q4.** You want to claim that your four-stage pipeline is better than SFT-only. What is the minimum experiment you must run (and report) to support this claim?

A. Compare your model to GPT-4o on your benchmark
B. Run an ablation where you train with SFT-only (no DPO, no GRPO) on the same data and compare accuracy on the same benchmark
C. Show that your DPO loss decreases during training
D. Show that your GRPO reward increases during training

## Short Answer

**Q5.** Explain the purpose of the LoRA `alpha` parameter and why the convention `alpha = 2 × rank` is commonly used.

**Q6.** Your compute budget section reports 12.4 A100-GPU-hours at $1.40/hr for a total training cost of $17.36. A reader asks: "Why is this so cheap compared to published models that cost $1M+?" Write a 3-sentence response suitable for including in a footnote.

**Q7.** You ran the GRPO stage with K=8 samples per prompt. Explain what happens to training quality and compute cost if you increase K to 16.

## Deep Scenario

**Q8.** Your DPO section reports: "DPO training ran for 800 steps with β=0.1, achieving a reward margin of 0.23." A senior researcher at your company reviews the draft and says: "The reward margin of 0.23 is meaningless without context. Also, how do you know DPO improved over SFT? You need ablation numbers here."

Write the revised DPO subsection (150–200 words) that addresses both criticisms by: (a) explaining what reward margin of 0.23 means in concrete terms, (b) reporting the accuracy of the SFT-only checkpoint vs the DPO checkpoint on your benchmark, and (c) adding one sentence explaining why you chose β=0.1 over other values.
