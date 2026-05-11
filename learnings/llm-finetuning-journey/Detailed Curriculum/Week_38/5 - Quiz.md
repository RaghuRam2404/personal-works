# Week 38 Quiz — Production QLoRA Fine-Tune

## Multiple Choice

**Q1.** Your 15K training run achieves eval loss 1.10 after epoch 1 and eval loss 1.08 after epoch 2. Should you run a third epoch?

A. Yes — loss is still decreasing, more epochs always help  
B. Only if the improvement (1.10 → 1.08) continues at the same rate in epoch 3 — monitor eval loss every 50 steps  
C. No — a 0.02 improvement is below measurement noise; the model has converged. Adding a third epoch risks overfitting.  
D. Yes — three epochs is always the standard for 15K examples

---

**Q2.** You compare your Week 38 model's 100-example exact match against GPT-4o-mini: your model scores 55%, GPT-4o-mini scores 71%. A colleague says: "Our model failed — GPT-4o-mini is better." Which is the most accurate interpretation?

A. Correct — our fine-tuned 7B model is worse; the project has failed  
B. Misleading — the goal for Phase 4 is to beat the base Qwen2.5-Coder-7B (5–15%), not GPT-4o. Beating GPT-4o is the Phase 6 goal. 55% represents a 4–10x improvement over base and significant progress.  
C. Misleading — exact match is not a valid metric; you should only compare on execution correctness  
D. Correct — the 7B model cannot compete with GPT-4o-mini because it is too small

---

**Q3.** Your training loss curve shows a spike from 0.7 to 2.1 at step 450, then recovers to 0.8 by step 500. What is the most likely cause, and what should you add to prevent it?

A. A corrupted evaluation batch; add `dataloader_drop_last=True`  
B. A single training batch with an unusually long or malformed example that caused a large gradient; add `max_grad_norm=1.0` to clip gradients  
C. The LR scheduler's first warmup step; add longer warmup  
D. The model checkpointing caused a brief numerical error; ignore it

---

**Q4.** After training, you call `model.save_pretrained("./postgres-sqlcoder-7b-v1")` on your Unsloth/PEFT model. The saved directory contains `adapter_model.safetensors` (50MB) and `adapter_config.json`. A user reports they cannot run inference because the model is missing the base model weights. What is the correct response?

A. You need to re-save using `merge_and_unload()` to include base model weights  
B. Explain that the adapter must be loaded onto the base model: `model = PeftModel.from_pretrained(base_model, adapter_path)`  
C. The save was incomplete; re-run with `push_to_hub` instead of `save_pretrained`  
D. The user needs to download Qwen2.5-Coder-7B separately and then load the adapter

---

**Q5.** You run error analysis on the 100 held-out examples. Results: 45% of errors are "wrong column referenced." What dataset addition would most directly address this failure mode?

A. Add more examples with simple SELECT queries (no complex features)  
B. Add more examples that include multiple plausible column names in the schema and questions that specifically require choosing the correct column  
C. Add more window function examples  
D. Increase training epochs from 2 to 4

---

## Short Answer

**Q6.** Your Week 38 model scores 58% exact match on 100 examples. Exact match is a conservative metric because semantically equivalent SQL may differ syntactically (e.g., `WHERE id = 1` vs. `WHERE id = '1'`). If execution correctness is typically 10–20 percentage points higher than exact match, what would you estimate Week 38's execution correctness to be? And why does this matter for the Phase 4 gate criterion?

---

**Q7.** You discover that 8 of the 100 held-out test examples contain TimescaleDB-specific queries (using `time_bucket`). Your model scores 2/8 on these examples (25% exact match) but 58% on the full 100. What does this tell you about your Week 37 dataset, and what would you change for a v2 fine-tune?

---

**Q8.** A colleague proposes: "Instead of pushing just the adapter to HuggingFace, let's merge the adapter into the base model weights and push the full 14GB model. That way users don't need to load the base model separately." Evaluate this proposal — when is it the right choice and when is it wrong?

---

## Scenario

**Q9.** You are 30 minutes into your Week 38 training run. W&B shows: train loss has gone from 2.3 → 1.8 → 1.6 → 1.3 (healthy decrease), but eval loss went from 2.4 → 2.3 → 2.4 → 2.5 (rising after initial small drop). You are at step 200 of 625 planned steps. What do you do?
