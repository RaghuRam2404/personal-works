# Week 35 Quiz — Hyperparameter Tuning for SFT/LoRA

## Multiple Choice

**Q1.** You have a 7B model, 24GB VRAM, and 10K training examples. You want to pick the LoRA learning rate. Which of the following is most likely to be the best starting point?

A. 1e-3 (fast convergence)  
B. 2e-4 (standard LoRA LR based on empirical consensus)  
C. 1e-6 (conservative, avoids overfitting)  
D. The same as the pretraining LR for this model (typically 3e-4 to 1e-3 for 7B models)

---

**Q2.** You run two training runs: Run A with LR=5e-5 for 3 epochs, Run B with LR=2e-4 for 1 epoch. Both have the same eval loss at their respective end points. Run B trained 3x faster. Which run has a better training setup for production use?

A. Run A — more epochs means more thorough training  
B. Run B — same quality, 3x faster, and fewer epochs mean less risk of overfitting  
C. Run A — more epochs provide more stability in the model's outputs  
D. Both are identical; training setup choice does not matter if eval loss is equal

---

**Q3.** You increase effective batch size from 16 to 64 (by increasing gradient accumulation steps from 4 to 16). According to the linear scaling rule, how should you adjust the learning rate?

A. Multiply LR by 4 (linear scaling with batch size ratio)  
B. Multiply LR by 2 (square root scaling: sqrt(64/16) = 2)  
C. Keep LR the same — batch size does not affect optimal LR  
D. Divide LR by 4 (larger batches need smaller LR to avoid overshooting)

---

**Q4.** What does `warmup_ratio=0.1` mean in `SFTConfig`, and what problem does it prevent?

A. 10% of parameters are randomly dropped at the start to prevent memorization  
B. The learning rate linearly increases from 0 to the target LR over the first 10% of training steps, preventing gradient explosions when the LoRA adapters are still random noise at initialization  
C. 10% of training examples are used for validation instead of training  
D. The model is frozen for the first 10% of training steps, then unfrozen

---

**Q5.** You run a sweep and find that rank 16 and rank 32 have identical eval loss on a 1K example dataset. You want to make a final decision for your 15K dataset (Week 38). What additional information would change your decision toward rank 32?

A. Nothing — identical eval loss means they are equivalent for all dataset sizes  
B. If your SQL task requires complex multi-join queries and subqueries that may need higher expressiveness, rank 32 might generalize better with more data  
C. If rank 32 runs 2x faster due to better parallelism  
D. If the W&B parallel coordinates show rank 32 has lower train loss

---

## Short Answer

**Q6.** Explain the interaction between learning rate and LoRA's alpha/rank scaling factor. If you keep LR=2e-4 but change rank from 16 to 64 (with alpha=2×rank for both), what effectively changes about the LoRA adapter training? What should you adjust to maintain the same effective update magnitude?

---

**Q7.** You are fine-tuning with 10K examples and observe: train loss decreasing from 2.1 to 0.4, eval loss decreasing from 2.1 to 0.8, then eval loss rising to 1.2 at the final epoch. Describe exactly what happened in training and what three changes you would make for the next run.

---

**Q8.** Sebastian Raschka's experiments show that covering all linear layers (q,k,v,o,gate,up,down) outperforms only q and v at the same rank. However, covering all layers with rank 32 uses the same total trainable parameters as covering only q and v with rank 128. Design the experiment you would run to determine which is better for your SQL task.

---

## Scenario

**Q9.** Your manager asks you to "tune all hyperparameters" for your SQL fine-tuning model before the Week 38 sprint. You have 3 hours of A100 time remaining. Design a practical hyperparameter search strategy: what do you sweep, in what order, and how do you allocate compute? Be specific about the number of runs and time per run.
