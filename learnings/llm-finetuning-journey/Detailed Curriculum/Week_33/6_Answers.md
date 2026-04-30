# Week 33 Quiz Answers

## Q1 — Answer: B

**Answer:** B. Frozen — no gradients, no updates.

**Why:** In QLoRA, the NF4 weights are loaded with `requires_grad=False` (handled automatically by bitsandbytes + peft). The forward pass dequantizes them to BF16 for computation, but this operation is not in the autograd computation graph — there is no gradient path to the original NF4 storage. Only the LoRA adapter matrices A and B are in the computation graph and receive gradients.

**Why others are wrong:**
- A: There is no separate LR for base weights — they have no LR because they have no gradients.
- C: This describes the compute path (dequantize for the forward pass), but re-quantization after computation does not happen — the quantized weights stay in NF4 storage throughout training.
- D: 8-bit AdamW is used for the LoRA optimizer states, not for updating the NF4 base.

---

## Q2 — Answer: B

**Answer:** B. The dtype used for matrix multiplication after dequantization.

**Why:** When a linear layer does its forward pass, bitsandbytes dequantizes the NF4 weights on the fly to BF16, performs the matmul in BF16, and returns BF16 results. The `compute_dtype` controls the precision of this on-the-fly matmul. Using BF16 (not FP16) prevents overflow and maintains numerical stability. The NF4 storage is unchanged — only the temporary compute buffer uses BF16.

---

## Q3 — Answer: B

**Answer:** B. Activation memory.

**Why:** The 7B NF4 model takes ~4.5GB. The LoRA adapters and their gradients + optimizer states add ~0.5–1GB. That leaves ~10–11GB for activations on a 16GB T4. At batch size 4, sequence length 512, with 32 transformer layers in a 7B model, activation memory can easily exceed 10GB — causing OOM. Fix: enable `gradient_checkpointing=True` (recomputes activations during backward, saving ~60–70% activation memory) or reduce batch size to 1–2.

---

## Q4 — Answer: B

**Answer:** B. Fails or loads incorrectly — `./adapter` contains only the adapter, not the full 7B model.

**Why:** `model.save_pretrained()` on a PeftModel saves only `adapter_model.safetensors` and `adapter_config.json`. `AutoModelForCausalLM.from_pretrained("./adapter")` looks for a full model checkpoint (config + weights) and either raises an error or partially loads. To use the adapter, a colleague must: (1) load the base model separately, (2) wrap with `PeftModel.from_pretrained(base, "./adapter")`.

---

## Q5 — Answer: B

**Answer:** B. Reduces optimizer state memory; paging allows CPU overflow.

**Why:** Standard 32-bit AdamW stores two fp32 tensors (first and second moment) per trainable parameter. For 42M LoRA parameters: 42M × 4 × 2 = 336MB. 8-bit AdamW (using bitsandbytes' 8-bit optimizer) quantizes these moment tensors to 8-bit, reducing to 42M × 1 × 2 = 84MB — 4x savings. Paging further allows these states to spill to CPU RAM if GPU memory is pressured, preventing OOM during long training runs.

---

## Q6 — Short Answer

**NF4 base:** The 7B weights stored in NF4 use ~4.5GB instead of 14GB (BF16) or 28GB (FP32). This is the dominant saving. The weights are frozen, so no gradient or optimizer state is needed for them.

**LoRA adapters:** Only ~42M parameters are trainable (at rank 16). In BF16: 42M × 2 bytes = 84MB. Gradients: 84MB. Total for the adapters themselves: ~170MB vs. ~84GB for full SFT (model + gradients + optimizer states for 7B parameters).

**Paged optimizer:** 8-bit AdamW quantizes the optimizer states for the 42M trainable params to ~84MB (vs. 336MB for fp32 Adam). Paging allows these states to live on CPU when VRAM is pressured. Combined: QLoRA total ≈ 4.5GB (model) + 0.5GB (adapters + optimizer) + 4–8GB (activations) ≈ 9–13GB. Full SFT would be: 14GB (weights) + 14GB (gradients) + 56GB (AdamW states) + activations ≈ 80–100GB.

---

## Q7 — Short Answer (5 hypotheses, ranked)

1. **Learning rate too low for the current training phase.** After a rapid initial drop, cosine LR schedule may have decayed too aggressively. By step 200, LR may be very small, causing near-zero updates. Check your LR schedule — the minimum LR should not be below 1e-5 for LoRA.

2. **Dataset fully exploited — model needs more diverse data.** 5K examples of SQL at max sequence 512 may not contain enough variety for the model to continue improving on this objective. More data (10K–15K) would likely break the plateau.

3. **Eval set is too similar to training set.** If eval loss also plateaed at the same value as train loss, the plateau may be the model's capacity limit for this dataset — it has learned what it can. This is not a bug but a signal to add data.

4. **Gradient checkpointing introducing silent numerical differences.** In rare cases, gradient checkpointing implementations can produce slightly different gradients than without it, causing training dynamics to stall. Try disabling it temporarily to verify.

5. **Packing creating cross-example attention.** With `packing=True` and no attention masking between packed examples, the model may be attending across example boundaries, introducing noise. Try `packing=False` to see if the plateau persists.

---

## Q8 — Short Answer

The most important additional evaluation: **execution correctness on a Postgres instance** — not exact match. Exact match is a conservative lower bound on quality: two different SQL queries can return identical rows (e.g., `SELECT name FROM t WHERE id=1` vs. `SELECT t.name FROM t WHERE t.id=1`). 42% exact match may correspond to 55–65% execution correctness. Conversely, some generated SQL may pass exact match but fail to execute (syntax error in a clause).

Run the 100 test examples through a PostgreSQL Docker container, execute both the expected SQL and the generated SQL, and compare result sets. This is the Week 39 eval harness. Your decision to deploy should be based on execution correctness, not exact match.

---

## Q9 — Scenario Answer

The concern is understandable but based on a misunderstanding of where gradients flow in QLoRA.

First: the NF4 weights do not participate in the backward pass at all. In PyTorch's autograd, a tensor only receives gradients if it has `requires_grad=True` and is part of the computation graph. The NF4 weights are loaded with `requires_grad=False`. During the forward pass, bitsandbytes dequantizes them on the fly to BF16 for the matmul, but this dequantization is performed outside the autograd graph (it is a "detached" operation). The loss gradient flows through the BF16 output of the matmul back to the LoRA matrices — not to the NF4 weights.

Second: there is a real but indirect effect. The NF4 quantization error affects the forward pass outputs — the activations that feed into the LoRA path are slightly different from what a BF16 base would produce. The LoRA matrices therefore learn to correct not just for the task alignment but also for some quantization artifacts. However, the adapters are in BF16 and receive accurate gradients — the issue is the quality of the signal (activations from a quantized base), not the gradient computation itself.

Third: the QLoRA paper empirically validates that this effect is small. Their benchmark results show QLoRA fine-tuned models match or exceed fully fine-tuned 16-bit models on standard NLP benchmarks at equivalent dataset sizes. The quantization noise in activations is dominated by the task signal for realistic training set sizes.
