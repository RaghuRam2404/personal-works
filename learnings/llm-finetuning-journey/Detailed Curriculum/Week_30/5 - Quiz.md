# Week 30 Quiz — LoRA: Math and Intuition

## Multiple Choice

**Q1.** You are applying LoRA with rank r=16, alpha=32 to a linear layer with d_in=2048, d_out=2048. How many trainable parameters does this LoRA adapter add?

A. 16 × 2048 = 32,768  
B. 16 × (2048 + 2048) = 65,536  
C. 2048 × 2048 = 4,194,304  
D. 32 × (2048 + 2048) = 131,072

---

**Q2.** Why is `lora_B` initialized to zero at the start of training?

A. Zero initialization prevents gradient explosion during the first backward pass  
B. It ensures that at step 0 the LoRA model behaves identically to the pretrained model — the adapter starts as a no-op  
C. It forces lora_A to learn first, which speeds up convergence  
D. It is a numerical stability trick similar to layer normalization epsilon

---

**Q3.** After LoRA training is complete, what is the most efficient way to deploy the model for inference?

A. Keep lora_A and lora_B separate and add their output at every forward pass  
B. Serialize lora_A and lora_B alongside the frozen model weights  
C. Merge lora_A and lora_B into W by computing W += (B @ A) * (alpha / r), then discard A and B  
D. Apply quantization to lora_A and lora_B before serving

---

**Q4.** The scaling factor `alpha / r` in the LoRA forward pass serves what purpose?

A. It normalizes the LoRA output to have unit variance  
B. It decouples the learning rate sensitivity from the rank hyperparameter, so changing r without changing alpha maintains a consistent update magnitude  
C. It prevents the LoRA matrices from exceeding the magnitude of the pretrained weight  
D. It is equivalent to weight decay applied to lora_A and lora_B

---

**Q5.** You fine-tune Qwen2.5-7B with LoRA rank 8 applied only to q_proj and v_proj. A colleague applied LoRA rank 8 to all linear layers (q, k, v, o, gate, up, down). At the same number of gradient steps and learning rate, which is likely to reach lower validation loss on a 5K SQL dataset?

A. Only q_proj and v_proj — fewer parameters means less overfitting  
B. All linear layers — covering more of the model gives the optimizer more directions to improve  
C. They will be identical — rank 8 is rank 8 regardless of which layers  
D. Only q_proj and v_proj — the original LoRA paper showed k and v are sufficient

---

## Short Answer

**Q6.** Derive the parameter count formula `r × (d_in + d_out)` from first principles, showing where each term comes from (matrix A dimensions, matrix B dimensions).

---

**Q7.** A colleague says: "LoRA cannot express full fine-tuning because it restricts delta_W to a rank-r matrix. If the true optimal delta_W has rank 50, a rank-16 LoRA will always be worse." Is this correct? Give a nuanced answer.

---

**Q8.** You apply LoRA rank 64 to all linear layers of Qwen2.5-7B. After training on 5K SQL examples for 1 epoch, you find the val loss is 0.3 but the model generates nonsensical SQL on every out-of-distribution example. What is happening, and what rank would you try next?

---

## Scenario

**Q9.** You are implementing LoRA from scratch for a production deployment. A junior engineer suggests: "Instead of initializing lora_B to zero, let's initialize both A and B with random values to make training faster — that way we get a non-zero delta_W from step 1." 

Explain why this is wrong with a concrete example. Show what would happen to training loss in the first 5 steps, and what the correct initialization scheme guarantees.
