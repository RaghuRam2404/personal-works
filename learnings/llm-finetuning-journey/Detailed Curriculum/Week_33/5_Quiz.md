# Week 33 Quiz — QLoRA

## Multiple Choice

**Q1.** In QLoRA training, the base model weights (7B parameters in NF4) are:

A. Trained with a lower learning rate than the LoRA adapters  
B. Frozen — they never receive gradients and are not updated during training  
C. Periodically dequantized to BF16 for gradient computation, then re-quantized  
D. Updated using 8-bit AdamW optimizer to save memory

---

**Q2.** You set `bnb_4bit_compute_dtype=torch.bfloat16` in `BitsAndBytesConfig`. What does this control?

A. The dtype used to store weights on disk  
B. The dtype used during matrix multiplication after on-the-fly dequantization from NF4  
C. The dtype of the LoRA adapter matrices A and B  
D. The dtype of the gradient accumulation buffer

---

**Q3.** Your QLoRA run on Colab Pro T4 (16GB VRAM) hits OOM after loading the 7B NF4 model (~4.5GB) and adding batch size 4. What is the most likely additional memory consumer causing OOM?

A. The LoRA adapter matrices (too large for T4)  
B. Activation memory for sequence length 512 with batch size 4, which can easily consume 6–10GB  
C. The paged optimizer has paged too much memory to GPU  
D. The tokenizer vocabulary embeddings are stored in GPU memory

---

**Q4.** After training, you call `model.save_pretrained("./adapter")` on your QLoRA PeftModel. Then a colleague tries to run: `model2 = AutoModelForCausalLM.from_pretrained("./adapter")`. What happens?

A. It loads correctly — `save_pretrained` saves the full model  
B. It fails or loads incorrectly because `./adapter` contains only the LoRA adapter weights, not the 7B base model  
C. It loads the NF4 quantized model without the LoRA adapters  
D. It loads the merged (full SFT) model

---

**Q5.** Why does QLoRA use `optim="paged_adamw_8bit"` rather than standard AdamW?

A. 8-bit AdamW converges faster than 32-bit AdamW for LoRA fine-tuning  
B. 8-bit AdamW reduces optimizer state memory for LoRA parameters from ~2× to ~0.5× the adapter size; paging allows states to overflow to CPU RAM under memory pressure  
C. paged_adamw_8bit automatically adjusts the learning rate for each LoRA layer  
D. Standard AdamW is incompatible with NF4 quantized base models

---

## Short Answer

**Q6.** Explain, in 3–4 sentences, why QLoRA can train a 7B model on a 24GB GPU when full SFT of a 7B model would require over 80GB. What specific memory savings does each component (NF4 base, LoRA adapters, paged optimizer) contribute?

---

**Q7.** Your QLoRA training loss is stuck at 1.2 after 500 steps (it started at 2.3 and dropped to 1.2 by step 200, then plateaued). The eval loss is also 1.2. Give 5 hypotheses ranked by likelihood for why training has plateaued.

---

**Q8.** You fine-tune Qwen2.5-Coder-7B with QLoRA rank 16 on 5K SQL examples. On your 100-example held-out test set, exact match is 42% vs. 8% for the base model. Your colleague says "great, 42% is good enough to ship." What is the most important additional evaluation you would run before agreeing to deploy?

---

## Scenario

**Q9.** You are presenting your QLoRA setup to a team of engineers. One engineer raises a concern: "The NF4 base model introduces quantization error. Won't this hurt the LoRA training because the gradients are computed through a lossy representation?"

Write a technically accurate 2–3 paragraph response explaining: (1) why the gradients do not flow through the NF4 weights, (2) what the actual source of quantization noise is in QLoRA, and (3) what the QLoRA paper shows empirically about the quality impact.
