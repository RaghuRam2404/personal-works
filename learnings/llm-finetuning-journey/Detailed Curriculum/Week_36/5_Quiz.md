# Week 36 Quiz — DoRA, RSLoRA, LoftQ

## Multiple Choice

**Q1.** DoRA decomposes a weight matrix W into magnitude and direction. In the DoRA forward pass, which component is updated by a LoRA-style low-rank matrix?

A. The magnitude vector m only  
B. The direction (V/||V||) only, approximated by a low-rank update BA  
C. Both magnitude and direction — m via a learned scalar vector, direction via BA  
D. Neither — DoRA only changes the initialization, not the update mechanism

---

**Q2.** You use RSLoRA with rank=64 and lora_alpha=16. What is the scaling factor applied to the LoRA output?

A. 16 / 64 = 0.25 (standard LoRA formula)  
B. 16 / sqrt(64) = 16 / 8 = 2.0 (RSLoRA formula)  
C. 64 / 16 = 4.0 (inverted for stability)  
D. sqrt(16 / 64) = 0.5 (geometric mean)

---

**Q3.** You are training with standard QLoRA (B=0 initialization) and find that at step 0, your model's loss is 4.5, while the BF16 base model's loss on the same data is 2.1. The quantized model loss is 2.4. What is the most likely cause of the additional gap between 2.4 and 4.5?

A. The NF4 quantization itself introduced errors, raising loss from 2.1 to 4.5  
B. The SFT training format (chat template) is different from the base model's pretraining format — the 4.5 is expected and represents the task difficulty, not quantization error  
C. Nothing is wrong — 4.5 is a normal starting loss for any SFT run  
D. The training dataset has formatting errors that inflate the loss

---

**Q4.** LoftQ initializes LoRA adapters by solving: B_init × A_init ≈ W_fp16 - W_nf4. What mathematical technique does it use to find B_init and A_init?

A. Gradient descent on the initialization loss  
B. Truncated SVD of the quantization error matrix (W_fp16 - W_nf4), keeping the top-r singular values  
C. Random initialization followed by 100 warm-up steps to converge to the quantization error  
D. Least-squares regression of the NF4 weights onto the FP16 weights

---

**Q5.** For your PostgreSQL text-to-SQL fine-tuning project (7B model, 5K examples, NF4 quantization), which LoRA variant should you try first, based on the empirical evidence from the DoRA paper?

A. LoftQ — because quantization error compensation is always the most important factor  
B. RSLoRA at rank 64 — always better than standard LoRA for 7B models  
C. DoRA at rank 16 — consistently outperforms standard LoRA on supervised fine-tuning tasks with modest additional overhead  
D. Standard LoRA at rank 16 — DoRA's benefits only appear at rank > 64

---

## Short Answer

**Q6.** Explain in 2–3 sentences why the RSLoRA scaling `alpha / sqrt(r)` is "rank-stabilized" compared to `alpha / r`. What happens to the LoRA output magnitude as rank increases under each formula (keeping alpha fixed)?

---

**Q7.** You compare DoRA and standard LoRA on your SQL dataset. DoRA achieves 48% exact match vs. LoRA's 45% on the held-out 100-example test. Your manager asks: "Should we always use DoRA going forward?" What factors would you consider before recommending DoRA as the default?

---

**Q8.** A colleague says: "Standard LoRA initializes B=0, which means the adapter starts as a no-op. LoftQ initializes with a non-zero B, which means the adapter starts with a non-zero output. Won't this destabilize training because the model no longer starts from the pretrained baseline?" Correct this misunderstanding.

---

## Scenario

**Q9.** You are preparing for the Week 38 15K example sprint. Based on your Week 36 experiments, you found: DoRA (rank 16) achieves eval loss 1.15, standard LoRA (rank 16) achieves 1.21, RSLoRA (rank 64) achieves 1.13. 

Considering: training time (DoRA adds ~5% overhead; RSLoRA rank 64 adds ~160M trainable params vs. 42M for rank 16), VRAM budget (A100 40GB), and Week 38 dataset size (15K), write a justified recommendation for which configuration to use in Week 38.
