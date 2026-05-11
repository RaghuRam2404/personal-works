# Week 31 Quiz — peft, Target Modules, Rank Sweeps

## Multiple Choice

**Q1.** You run `model.save_pretrained("./my-lora-adapter")` on a PEFT LoRA model. What is saved to disk?

A. The full fine-tuned model weights (14GB for a 7B model)  
B. Only the LoRA adapter matrices A and B, plus the adapter configuration JSON  
C. The base model weights merged with the LoRA update  
D. Only the LoraConfig JSON file without any weight tensors

---

**Q2.** You are applying LoRA to Qwen2.5-Coder-1.5B with `target_modules=["q_proj", "v_proj"]`. A colleague says you should also include `"gate_proj"`, `"up_proj"`, `"down_proj"`. What is the strongest empirical argument for their position?

A. The MLP layers are responsible for grammar and syntax in transformer language models; adapting them directly improves SQL structure  
B. Sebastian Raschka's experiments consistently show that covering all linear layers outperforms attention-only LoRA at the same rank  
C. The peft library requires at least 5 target modules to function correctly  
D. MLP layers have larger d_out dimensions, so LoRA on MLPs has higher parameter efficiency per layer

---

**Q3.** You run a rank sweep with `lora_alpha = 2 * r` for all runs (so scaling = 2.0 is constant). After training on 5K examples, you get:

| Rank | Eval Loss |
|---|---|
| 8 | 1.42 |
| 16 | 1.28 |
| 32 | 1.25 |
| 64 | 1.38 |

What is the most likely explanation for why rank 64 performs worse than rank 32?

A. Rank 64 has a bug in the peft implementation for large ranks  
B. Rank 64 has more trainable parameters (160M+ for a 7B model), causing overfitting on the 5K dataset  
C. The scaling factor 2.0 is too large for rank 64, causing gradient explosion  
D. Rank 64 requires a different learning rate scheduler

---

**Q4.** You want to load a previously saved LoRA adapter onto a base model for inference. Which code sequence is correct?

A. `model = AutoModelForCausalLM.from_pretrained("./my-lora-adapter")`  
B. `model = PeftModel.from_pretrained(base_model, "./my-lora-adapter")`  
C. `model = get_peft_model(base_model, LoraConfig.from_pretrained("./my-lora-adapter"))`  
D. `base_model.load_state_dict(torch.load("./my-lora-adapter/adapter_model.safetensors"))`

---

**Q5.** You have 24GB VRAM and want to fine-tune Qwen2.5-Coder-7B with LoRA rank 16 on all linear layers. The base model in bfloat16 requires ~14GB. With LoRA rank 16, approximately how much additional VRAM is needed for the adapter gradients and optimizer states?

A. ~56GB (same as full SFT for the 7B model)  
B. ~0.5–1GB (adapter matrices are tiny; only A and B have gradients/optimizer states)  
C. ~14GB (one copy of the model for gradients)  
D. ~28GB (two copies of the model for forward and backward pass)

---

## Short Answer

**Q6.** You are fine-tuning Qwen2.5-Coder-7B with LoRA rank 16 on 10K SQL examples. After training, you push the adapter to HuggingFace. A user reports that when they download and run the model, the outputs look like the base model (random text, no SQL). What are two possible causes, and how would you diagnose each?

---

**Q7.** Explain what `bias="none"` means in `LoraConfig` and why it is the default. When might `bias="all"` be appropriate?

---

**Q8.** You run your rank sweep and find that both rank 16 and rank 64 converge to the same eval loss (1.28). What does this tell you about the intrinsic dimensionality of the SQL fine-tuning task relative to your 1.5B model? What practical decision does this inform?

---

## Scenario

**Q9.** A product manager asks: "We have a 7B model, 24GB GPU, and 10K training examples. Pick a LoRA rank, alpha, and list of target_modules. Justify each choice."

Write a complete answer with specific values and a sentence of justification for each.
