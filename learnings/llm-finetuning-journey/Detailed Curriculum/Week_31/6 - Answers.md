# Week 31 Quiz Answers

## Q1 — Answer: B

**Answer:** B. Only LoRA adapter matrices and configuration.

**Why:** `model.save_pretrained()` on a PeftModel saves only `adapter_model.safetensors` (containing lora_A and lora_B for all target layers) and `adapter_config.json` (containing the LoraConfig). The base model weights are not duplicated. This is the key advantage: a rank-16 adapter for Qwen2.5-7B is ~100MB rather than 14GB. Users download the base model separately and apply the adapter on load.

**Why others are wrong:**
- A: That would require duplicating the 14GB base model — defeating the purpose of PEFT.
- C: Saving merged weights requires calling `model.merge_and_unload()` first, then `save_pretrained()` on the merged model.
- D: The config alone without weights would be useless for inference.

---

## Q2 — Answer: B

**Answer:** B. Empirical evidence from Raschka's experiments.

**Why:** The most rigorous argument is empirical: Raschka's systematic comparison across multiple fine-tuning tasks found that covering all linear layers (q, k, v, o, gate, up, down) consistently outperformed covering only attention projections at the same rank. The theoretical explanation (option A) has some merit but is oversimplified — MLP layers do more than grammar and syntax.

**Why others are wrong:**
- C: peft has no minimum target_modules count.
- D: Parameter efficiency is measured per total adapters, not per layer size. Adapting large MLP layers is less efficient per parameter than small attention layers.

---

## Q3 — Answer: B

**Answer:** B. Rank 64 overfits on 5K examples.

**Why:** With `lora_alpha = 2r`, rank 64 adds ~160M trainable parameters to a 7B model (about 2.2% of total). On 5K examples, this capacity allows memorization of training examples. The optimizer drives training loss very low (potentially 0.3–0.5) at the cost of generalization. Rank 32 (80M trainable) hits the sweet spot — enough capacity to capture SQL patterns without memorizing the training set.

---

## Q4 — Answer: B

**Answer:** B. `PeftModel.from_pretrained(base_model, adapter_path)`

**Why:** Loading a LoRA adapter requires: (1) loading the base model separately (with its own `from_pretrained`), then (2) wrapping it with `PeftModel.from_pretrained` which reads `adapter_config.json` to reconstruct the LoRA structure and loads weights from `adapter_model.safetensors`.

**Why others are wrong:**
- A: `AutoModelForCausalLM.from_pretrained` on an adapter directory fails because it's not a full model checkpoint.
- C: This would create new random LoRA matrices using only the config, not load trained weights.
- D: State dict loading would fail due to key mismatches between the adapter structure and base model.

---

## Q5 — Answer: B

**Answer:** B. ~0.5–1GB for adapter gradients and optimizer states.

**Why:** LoRA rank-16 on all linear layers of a 7B model has approximately 40M trainable parameters (about 0.25B model equivalent). In bfloat16: 40M params × 2 bytes = 80MB for the adapter itself. Gradients (also bfloat16): 80MB. AdamW first/second moments (fp32): 40M × 4 × 2 = 320MB. Total adapter overhead: ~480MB. This is why LoRA fine-tuning on a 7B model fits comfortably in 24GB VRAM: 14GB (model) + 0.5GB (adapters) + 1–2GB (activations, batch) ≈ 16–17GB.

---

## Q6 — Short Answer

**Cause 1: User did not apply the chat template.** If the user prompts the model with raw text instead of the ChatML format (`<|im_start|>user...`), the model sees an unfamiliar input distribution and falls back to base model behavior. Diagnosis: print the model's raw tokenized input at inference time and check for `<|im_start|>` tokens.

**Cause 2: Adapter weights not loaded correctly — user loaded the base model without the adapter.** If the user called `AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-7B")` without `PeftModel.from_pretrained`, they are running the base model. Diagnosis: call `type(model)` — should be `PeftModelForCausalLM`, not `Qwen2ForCausalLM`. Or check `model.print_trainable_parameters()` — should show LoRA params as trainable.

---

## Q7 — Short Answer

`bias="none"` means that bias vectors (the additive term in `y = Wx + b`) are excluded from LoRA adaptation — they remain frozen at their pretrained values. This is the default because: (1) biases are small (one scalar per output dimension, not a matrix), so adapting them adds minimal expressiveness while complicating the adapter format; (2) the LoRA paper found no meaningful improvement from adapting biases; (3) not adapting biases keeps the adapter format clean — only A and B matrices.

`bias="all"` might be appropriate if you observe systematic output shifts that cannot be corrected by the weight-only LoRA adaptation — for example, if the model consistently over-generates certain tokens or under-generates others. In practice, this is rare for SFT tasks.

---

## Q8 — Short Answer

If rank 16 and rank 64 converge to the same eval loss, the intrinsic dimensionality of the SQL fine-tuning task for this model-dataset combination is at most rank 16. The extra dimensions provided by rank 64 contribute nothing because all the useful gradient signal fits within a 16-dimensional subspace of the weight update space.

Practical decision: use rank 16. It trains faster (fewer parameters to update), uses less memory, and generalizes identically to rank 64. This also suggests that even rank 8 might perform similarly — worth checking if memory is a constraint.

---

## Q9 — Scenario Answer

**Rank: 16**
Justification: 10K examples is a moderate dataset for a 7B model. Rank 16 provides ~40M trainable parameters (0.25% of total), which is sufficient to capture SQL generation patterns without overfitting. Rank 64 risks memorization on 10K examples; rank 8 risks underfitting on a complex multi-table SQL task. If empirical results show rank 16 underfitting (eval loss plateaus high), increase to 32.

**Alpha: 32** (= 2 × rank)
Justification: Sets the LoRA scaling factor to 2.0, which keeps the adapter update at a meaningful magnitude relative to the pretrained weights without being too aggressive. This is the standard default from the LoRA paper and Raschka's empirical findings.

**Target modules: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`**
Justification: All linear layers. For SQL fine-tuning, the MLP layers (gate/up/down) store the model's factual and syntactic SQL knowledge; adapting only attention leaves half the model's relevant representations un-updated. At rank 16, covering all 7 layer types still uses less than 1% of 7B parameters.

**VRAM estimate:** 14GB (model bf16) + 0.5GB (adapters) + 2GB (activations at batch 4, seq 512) + overhead ≈ 17GB — fits in 24GB with headroom.
