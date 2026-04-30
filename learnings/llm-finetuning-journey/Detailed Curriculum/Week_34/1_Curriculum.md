# Week 34 — Unsloth: The Speed Unlock

## Learning Objectives

By the end of this week, you will be able to:

- Explain the core optimizations Unsloth applies to make LoRA/QLoRA training 2–5x faster
- Re-run your Week 33 QLoRA training with Unsloth and empirically verify the speedup
- Compare training time, VRAM usage, and final model quality between vanilla HuggingFace QLoRA and Unsloth
- Configure Unsloth for Qwen2.5-Coder-7B fine-tuning
- Understand which optimizations Unsloth provides and what their limitations are

---

## Concepts

### 1. Why Vanilla QLoRA is Slow

Your Week 33 training ran on A100 and took 25–45 minutes for 5K examples. For 15K examples (the Week 38 sprint), that's ~75–135 minutes. Several sources of inefficiency in vanilla HuggingFace QLoRA:

1. **Flash Attention 2 not used by default.** Standard scaled dot-product attention stores O(n²) attention matrices in memory and processes them sequentially. Flash Attention 2 uses tiling to compute attention in O(1) memory with better hardware utilization.

2. **LoRA backward pass is unoptimized.** The standard peft implementation computes `x @ A.T @ B.T` with two separate matmuls, each with their own memory allocation.

3. **Activation memory overhead.** Without custom kernels, intermediate activations are stored in full even when gradient checkpointing is enabled.

4. **Tokenization bottleneck.** Standard DataLoader tokenizes/processes examples on-the-fly during training.

### 2. What Unsloth Does

Unsloth (Daniel Han, Tim Han) addresses these bottlenecks with custom Triton/CUDA kernels:

**Fused LoRA computation:** Unsloth fuses the forward and backward passes of the LoRA adapters into a single kernel that avoids redundant memory reads. The `lora_out = x @ A.T @ B.T` is computed in one pass rather than two.

**Custom RoPE embedding kernel:** Rotary position embeddings (used by Qwen2.5, Llama, Mistral) are precomputed and cached via a custom kernel that avoids repeated computation.

**Triton-based Flash Attention integration:** Unsloth integrates Flash Attention 2 via Triton kernels optimized specifically for the LoRA training pattern.

**VRAM reduction:** Unsloth's kernels reduce peak VRAM usage by 40–60% compared to vanilla HuggingFace training on the same model and batch size. This allows larger batch sizes on the same hardware.

The combined effect: 2–5x faster training with 40–60% less VRAM.

### 3. The Unsloth API

Unsloth provides its own model loading functions that wrap HuggingFace models with their optimized kernels:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-Coder-7B",
    max_seq_length=2048,
    dtype=None,               # Auto: BF16 on Ampere+
    load_in_4bit=True,        # QLoRA by default
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,           # Unsloth docs recommend 0 dropout for max speed
    bias="none",
    use_gradient_checkpointing="unsloth",  # Unsloth's custom GC
    random_state=42,
    use_rslora=False,         # RSLoRA is available — Week 36
)
```

Then pass to the same `SFTTrainer` from Week 33. The rest of the training loop is identical.

**Key difference from Week 33:** Use `use_gradient_checkpointing="unsloth"` instead of `gradient_checkpointing=True` in SFTConfig. Unsloth's GC implementation is more memory-efficient than HuggingFace's default.

### 4. Measuring the Speedup

When comparing vanilla QLoRA vs. Unsloth, measure:

1. **Steps per second** (reported by SFTTrainer's progress bar and W&B)
2. **Peak VRAM** (`torch.cuda.max_memory_allocated()` at the end of first epoch)
3. **Training time** (wall clock for same number of steps)
4. **Final loss** at same step count (quality should be equivalent)

A legitimate speedup should show 2x+ in steps/second without loss quality degradation.

### 5. Unsloth Limitations

Unsloth's optimizations are powerful but have constraints:

- **GPU support:** Requires NVIDIA GPU with CUDA. Does not run on Mac MPS or AMD ROCm.
- **Model support:** Supports a curated list of architectures (Qwen2.5, Llama, Mistral, Gemma, etc.). Not all HuggingFace models are supported.
- **Dropout:** Unsloth recommends `lora_dropout=0` for maximum speed. If dropout is needed for regularization, there is a small speed cost.
- **Custom modifications:** If you need to modify the model architecture (e.g., custom attention patterns), Unsloth's wrapping may conflict.
- **Version pinning:** Unsloth is updated frequently. Pin the version in your requirements file.

### 6. Unsloth's Recommendation for This Curriculum

From Week 34 onward, use Unsloth for all single-GPU training runs. The speedup is significant and the API is almost identical to vanilla HuggingFace. In Phase 5–6, when you use RunPod (multi-GPU), you will revisit whether Unsloth or DeepSpeed is more appropriate.

The recommended stack going forward:
- **Unsloth** for single-GPU Colab/RunPod training (Weeks 34–52)
- **DeepSpeed ZeRO-3** for multi-GPU distributed training (Phase 6)
- **VLLM** for batch inference serving (Phase 5–6)

---

## Connections

**Builds on:** Week 33 QLoRA setup — identical training loop, just faster. This week is a direct speedup measurement on the same task.

**Needed for:** Week 38 (the 15K domain sprint will use Unsloth). All future training runs in Phase 5.

---

## Common Misconceptions / Pitfalls

- **"Unsloth changes the model quality."** No — the mathematical operations are identical; only the kernel implementation is optimized. The final model quality should be equivalent at the same step count and hyperparameters.
- **"Unsloth replaces peft."** No — Unsloth wraps peft's LoRA infrastructure with faster kernels. The adapter format is compatible with peft for loading/saving.
- **"lora_dropout=0 is safe for all datasets."** Dropout provides regularization. For small datasets (<5K) where overfitting is a risk, you may want `lora_dropout=0.05` even at a small speed cost.
- **"Unsloth works on Mac."** No — requires CUDA. Run it on Colab Pro.

---

## Time Allocation (6–8 hrs)

| Activity | Time |
|---|---|
| Read Unsloth README and relevant blog posts | 1h |
| Convert Week 33 training script to Unsloth | 1h |
| Run vanilla QLoRA (from Week 33 script) and record baseline metrics | 1.5h |
| Run same training with Unsloth, record metrics | 1.5h |
| Write comparison report | 1h |
| Commit to GitHub | 30m |
