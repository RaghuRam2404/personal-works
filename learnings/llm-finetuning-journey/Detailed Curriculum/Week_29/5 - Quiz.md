# Week 29 Quiz — Full SFT on a Tiny Model

## Multiple Choice

**Q1.** You are training `Qwen2.5-0.5B` with `SFTTrainer` and notice that the model is learning to predict the prompt tokens as well as the response tokens. Which of the following is most likely the cause?

A. The model's learning rate is too high  
B. The `dataset_text_field` is set correctly but `add_generation_prompt=True` was passed during preprocessing  
C. The chat template was not applied, and examples are formatted as raw `(question, answer)` strings without role markers — so SFTTrainer cannot detect where the prompt ends  
D. The model needs `gradient_checkpointing=True` to properly mask inputs

---

**Q2.** Your SFT training loss curve looks like: starts at 2.4, drops to 0.9 by step 200, then continues dropping to 0.05 by step 1000. Meanwhile, eval loss is 1.8 at step 200 and 2.5 at step 1000. What is happening?

A. Training is going well; low train loss means the model learned well  
B. The model is overfitting; eval loss rising while train loss falls is the classic signal — you should have stopped at step 200  
C. The eval dataset is corrupted; ignore eval loss  
D. The learning rate schedule is wrong; switch to linear decay

---

**Q3.** What memory is consumed by a 0.5B parameter model during full SFT with AdamW optimizer and fp16 precision?

A. ~0.5GB (one byte per parameter in fp16)  
B. ~1GB (two bytes per parameter in fp16)  
C. ~6GB (model weights 1GB + gradients 1GB + AdamW states 4GB)  
D. ~12GB (double-precision optimizer states dominate)

---

**Q4.** Which statement about `packing=True` in `SFTConfig` is correct?

A. It packs multiple LoRA adapters into a single model weight  
B. It concatenates multiple short training examples into a single sequence up to `max_seq_length`, improving GPU utilization  
C. It enables gradient checkpointing to reduce memory usage  
D. It applies byte-pair encoding to pack rare tokens

---

**Q5.** You push `Qwen2.5-0.5B` fine-tuned on 1K SQL examples to HuggingFace. A colleague loads it and complains the model outputs garbage. They ran: `output = model.generate(tokenizer.encode("What is the average salary?", return_tensors="pt"))`. What is the most likely problem?

A. The model was not properly fine-tuned  
B. The input was not formatted with the chat template; the model expects `<|im_start|>user` structure  
C. The tokenizer was not pushed to the Hub alongside the model  
D. `model.generate` does not work with fine-tuned models

---

## Short Answer

**Q6.** Explain in 3–4 sentences why full SFT (updating all parameters) is practical for `Qwen2.5-0.5B` but impractical for `Qwen2.5-7B` on a 16GB GPU. What is the memory bottleneck?

---

**Q7.** Your SFT model generates SQL that is syntactically correct but semantically wrong — for example, it writes `SELECT name FROM orders` when the correct answer is `SELECT name FROM customers`. What could cause this? Give two hypotheses.

---

**Q8.** You have 10K SQL training examples but only 1K are PostgreSQL-specific (using `generate_series`, `LATERAL`, `JSONB` operators). The other 9K are generic SQL (MySQL/SQLite style). Should you train on all 10K or just the 1K PostgreSQL examples? Justify your answer with reasoning about what SFT learns.

---

## Scenario

**Q9.** You are running your Week 29 training and the loss curve shows: steady decrease from 2.3 to 1.8 over the first 100 steps, then sudden spike to 8.5, then NaN. Training crashes.

Provide exactly 4 hypotheses for what caused the spike, ranked from most to least likely, with one concrete fix for each.
