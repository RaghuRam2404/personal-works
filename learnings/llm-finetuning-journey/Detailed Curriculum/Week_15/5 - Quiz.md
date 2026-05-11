# Week 15 Quiz — From-Scratch GPT-2 Reproduction

Calibration: mid-junior ML interview level. Questions cover production training details.

---

**Q1.** You are training GPT-2 124M with gradient accumulation: `grad_accum_steps=32`, `B=16`, `T=1024`. You forget to divide the loss by `grad_accum_steps`. What is the consequence?

A) No effect — gradient accumulation normalizes automatically  
B) The effective gradient is 32x larger than intended, causing instability or divergence in the first optimizer step  
C) The val loss is 32x worse than expected  
D) Training is 32x slower because gradients overflow  

---

**Q2.** `torch.autocast(device_type='cuda', dtype=torch.bfloat16)` is applied only during the forward pass. Why aren't the optimizer states (Adam's m and v) also stored in bfloat16?

A) PyTorch doesn't support bfloat16 optimizer states  
B) Adam's first and second moment accumulators need high precision to accumulate small updates faithfully over many steps; bfloat16's limited mantissa would cause updates to be quantized to zero  
C) Optimizer states are always in bfloat16 when autocast is active — this statement is incorrect  
D) The optimizer runs on CPU which doesn't support bfloat16  

---

**Q3.** You replace manual scaled dot-product attention with `F.scaled_dot_product_attention(q, k, v, is_causal=True)`. What benefit does this provide, and what do you no longer need to handle?

A) It automatically selects Flash Attention when available, avoiding materializing the O(T²) attention matrix in HBM; you no longer need to compute the causal mask manually  
B) It fuses the attention with the MLP, reducing memory transfers  
C) It replaces Q, K, V projections with a single fused kernel  
D) It automatically applies gradient checkpointing to the attention operation  

---

**Q4.** During training you observe that `grad_norm` (printed before clipping) is consistently hitting the 1.0 clip limit starting from step 100. What does this indicate?

A) The model is converging correctly — gradient clipping is expected throughout training  
B) The effective learning rate may be too high, or the loss landscape is very sharp; gradients are consistently large, which may slow convergence  
C) The gradient accumulation is not working — all gradients come from a single batch  
D) Weight tying is causing gradient explosion in the embedding layer  

---

**Q5.** HellaSwag accuracy for GPT-2 124M is ~29.5% vs. random baseline of 25%. You train a model and achieve 45% HellaSwag accuracy after the same number of steps. What are two possible explanations?

A) Your model is larger than 124M parameters, or your training data is higher quality  
B) Your tokenizer has a larger vocabulary and encodes text more efficiently  
C) You have a bug where the model sees the correct answer label during evaluation  
D) Both A and C are possible  

---

**Q6 (short answer).** Explain gradient accumulation in detail: what problem it solves, how it works mathematically, and what must be done to the loss before calling `loss.backward()` to keep it equivalent to training with a single large batch.

---

**Q7 (short answer).** You are training GPT-2 124M on a single A100 (40GB). Your target is `total_batch_size=524288` tokens. You set `B=16, T=1024`. How many micro-steps per logical batch? Now you want to increase `T` to 2048 to help the model learn longer-range dependencies. What happens to `grad_accum_steps`, and what is the memory implication?

---

**Q8 (scenario).** At step 0, your GPT-2 reproduction reports `train_loss=3.2`. You expect it to be ~10.82. What is the most likely bug, and how do you fix it?
