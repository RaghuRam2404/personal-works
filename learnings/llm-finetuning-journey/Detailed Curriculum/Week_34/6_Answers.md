# Week 34 Quiz Answers

## Q1 — Answer: C

**Answer:** C. Automatic model parallelism is NOT an Unsloth optimization.

**Why:** Unsloth is explicitly designed for single-GPU training. It does not implement model parallelism or data parallelism across GPUs. Its optimizations are all at the single-GPU kernel level: fused LoRA matmuls (A), custom gradient checkpointing (B), and custom RoPE kernels (D). For multi-GPU training, you would use DeepSpeed or FSDP.

---

## Q2 — Answer: B

**Answer:** B. Increase batch size to improve throughput and potentially quality.

**Why:** Going from 24GB to 14GB peak VRAM frees 10GB. At the current batch size 4, each example uses ~6GB of activation memory. The freed 10GB allows increasing batch size to 8–10 without exceeding 24GB. Larger effective batch size (with gradient accumulation) provides more stable gradients and can improve convergence, especially on noisy data like SQL. This is the correct leveraging of Unsloth's VRAM savings.

**Why A is wrong:** Reducing max_seq_length when you have freed memory is a waste of the optimization benefit.

---

## Q3 — Answer: B

**Answer:** B. `gradient_checkpointing=True` in SFTConfig.

**Why:** `use_gradient_checkpointing="unsloth"` in `get_peft_model` hooks Unsloth's custom GC implementation directly into the model. If you also set `gradient_checkpointing=True` in `SFTConfig`, `SFTTrainer` calls `model.gradient_checkpointing_enable()` which installs PyTorch's default GC, potentially overriding Unsloth's implementation or causing double-hookup errors. Use Unsloth's GC exclusively.

---

## Q4 — Answer: B

**Answer:** B. Within training variance — effectively equivalent.

**Why:** LLM fine-tuning loss curves have natural variance from gradient noise, data ordering, and floating-point non-determinism. A difference of 0.04 loss units at step 1000 is well within this variance range — if you reran vanilla QLoRA with a different random seed, you would likely see 0.03–0.08 variance. The correct conclusion is that Unsloth's kernel produces numerically equivalent results (not identical due to floating-point order of operations). Before concluding numerical differences, run 3 seeds per method and compute mean ± std.

---

## Q5 — Answer: B

**Answer:** B. Vanilla HuggingFace peft + bitsandbytes + SFTTrainer.

**Why:** The Week 33 QLoRA setup (peft + bitsandbytes + SFTTrainer) is the universal fallback for any model that Unsloth does not support. It works with any HuggingFace `AutoModelForCausalLM` model and any `nn.Linear` layer names. Unsloth adds speed on top of this foundation; removing Unsloth reverts to the solid but slower baseline. Axolotl is another popular training framework but does not replace Unsloth's kernels for all models.

---

## Q6 — Short Answer

Standard scaled dot-product attention computes the full attention matrix A = softmax(QK^T / sqrt(d_k)) of shape (batch, heads, seq_len, seq_len), which requires O(n²) memory. For `seq_len=2048`, this is 2048² = 4M entries per attention head per batch element — substantial memory at high batch sizes. Flash Attention 2 uses block tiling to compute the softmax in pieces, processing blocks of Q against blocks of K and V without materializing the full n×n matrix. The result is O(n) memory (proportional to sequence length, not its square) with better cache utilization on GPU SRAM. For `max_seq_length=2048` on a model with 32 attention heads, Flash Attention 2 reduces attention VRAM by ~32x compared to standard attention, directly enabling longer contexts and larger batch sizes.

---

## Q7 — Math Answer

With 15K examples and packing ratio 3 (3 examples per sequence):
- Number of packed sequences = 15,000 / 3 = 5,000 sequences
- Per epoch = 5,000 steps (at batch size 1; or 5000 / batch_size steps at larger batches)
- With gradient accumulation steps 4 and batch size 4: effective batch = 16
  - Steps per epoch = 5,000 / 4 = 1,250 optimizer steps
  - 2 epochs = 2,500 optimizer steps

At 3 steps/second (optimizer steps):
- Time = 2,500 / 3 ≈ 833 seconds ≈ 14 minutes

Cost: 14 minutes / 60 × $1.20/hr ≈ $0.28. Well within the $10 budget. (Include notebook startup, evaluation time: total maybe $0.50–1.00 for the full session.)

---

## Q8 — Short Answer

This is a bad idea. Unsloth requires a CUDA GPU and does not support Apple Silicon MPS or AMD ROCm. Running on Mac M3 Max will fail at import or at the first training step.

Recommended approach for local Mac development:
1. Use the vanilla QLoRA setup (peft + bitsandbytes) on CPU — bitsandbytes has CPU emulation for debugging (slow but functional for correctness checks)
2. Alternatively, run a tiny model (Qwen2.5-0.5B) with standard SFTTrainer on Mac MPS for local debugging
3. Then run Unsloth only on Colab Pro A100 for actual training

Local Mac = debugging with 0.5B model on MPS. Colab A100 = production training with Unsloth + 7B.

---

## Q9 — Scenario Answer

Given: 15K examples, packing ratio 3, batch size 4, gradient accumulation 4:
- Packed sequences per epoch: 15,000 / 3 = 5,000
- Steps per epoch: 5,000 / (4 × 4) = 312 optimizer steps → 2 epochs = 625 optimizer steps

At 3 steps/second: 625 / 3 ≈ 208 seconds ≈ 3.5 minutes of pure training. Including data loading, evaluation, checkpointing: total session ~10–20 minutes.

Cost: 20 min / 60 × $1.20/hr ≈ $0.40. Far under the $10 budget.

Decision: proceed immediately. The training is cheap and fast. If you want to be safe: run a 100-step smoke test first (~30 seconds) to verify everything works, then run the full 2 epochs. Budget for full Week 38 sprint (including iteration): ~$3–5 from your remaining credits.
