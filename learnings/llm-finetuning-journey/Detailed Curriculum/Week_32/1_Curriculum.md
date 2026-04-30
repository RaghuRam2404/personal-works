# Week 32 — Quantization Fundamentals

## Learning Objectives

By the end of this week, you will be able to:

- Explain the difference between FP32, FP16, BF16, INT8, INT4, NF4, and FP4 formats and their trade-offs
- Distinguish weight-only quantization from activation quantization
- Explain how GPTQ, AWQ, and LLM.int8() work at a conceptual level
- Quantize `Qwen2.5-Coder-1.5B` to 4-bit using `bitsandbytes` and measure the memory and quality trade-offs
- Build and present a comparison table: model size, VRAM, perplexity, inference speed across quantization formats

---

## Concepts

### 1. Why Quantization?

A 7B model in FP32 requires 7 × 4 = 28GB. In FP16/BF16: 14GB. In INT8: 7GB. In 4-bit: 3.5GB. Quantization is what makes inference (and with QLoRA, training) possible on consumer hardware.

The trade-off: lower precision = less memory = faster compute = potential accuracy loss. The art of quantization is finding where on this curve you can operate without unacceptable quality degradation.

### 2. Floating-Point Formats

**FP32 (float32):** 1 sign bit, 8 exponent bits, 23 mantissa bits. Maximum value: 3.4 × 10^38. This is the default PyTorch training format.

**FP16 (float16):** 1 sign, 5 exponent, 10 mantissa. Range: ±65504. Used for mixed-precision training. Can overflow for large gradients (a known issue).

**BF16 (bfloat16):** 1 sign, 8 exponent, 7 mantissa. Same exponent range as FP32 but lower precision. Numerically more stable than FP16 for training because the wider exponent range prevents overflow. Standard for modern LLM training on A100/H100.

**INT8:** 8-bit integer, range −128 to 127. No floating-point. To represent model weights, you need a scale factor: `W_float ≈ W_int8 × scale`. Quantization error = the difference between the float value and its int8 approximation.

**INT4:** 4-bit integer, range −8 to 7. Very aggressive compression. Requires careful calibration to avoid significant accuracy loss.

**NF4 (NormalFloat4):** Dettmers et al.'s key contribution in QLoRA. Instead of evenly spaced integer values, NF4 uses quantile-based data types: the 16 possible 4-bit values are placed at the quantiles of a standard normal distribution. This is optimal for normally distributed weights (which LLM weights typically are). NF4 achieves lower quantization error than INT4 for LLM weights.

**FP4:** 4-bit floating-point format. Fewer values than NF4, used in some NVIDIA hardware, but generally less accurate than NF4 for LLM weights.

### 3. Weight-Only vs. Activation Quantization

**Weight-only quantization:** Quantize the weight matrices; keep activations in full precision during inference. Simpler to implement; the primary source of memory savings. GPTQ, AWQ, and bitsandbytes 4-bit are all weight-only.

**Activation quantization:** Quantize both weights and activations. More challenging because activations have outliers (Dettmers found that ~0.1% of activations in large LLMs are massive outliers that dominate quantization error). LLM.int8() specifically addresses this.

### 4. LLM.int8(): Mixed-Precision Decomposition

Dettmers (2022) observed that activations in large LLMs have a systematic outlier problem: a small number of activation dimensions (often the same dimensions across all inputs) take values 100–1000x larger than typical values. Quantizing these with a single global scale factor causes enormous error.

LLM.int8()'s solution: for each matrix multiply, identify the outlier dimensions (threshold typically 6.0), extract them into a separate FP16 multiply, and compute the non-outlier dimensions in INT8. The result:

```
Y = Y_outlier_fp16 + Y_normal_int8
```

This gives near-FP16 quality with ~2x memory savings. It requires a CUDA-optimized implementation (in `bitsandbytes`). Available via `load_in_8bit=True`.

### 5. GPTQ: Post-Training Quantization with Second-Order Information

GPTQ (Frantar et al., 2022) is a post-training quantization method. Given a calibration dataset (a few hundred examples), it quantizes weights layer-by-layer, compensating for quantization error in each weight using a Hessian-based correction applied to the remaining weights.

Key idea: when you quantize one weight and introduce error, you can reduce the overall error by slightly adjusting the remaining unquantized weights in the same row (using inverse Hessian information). This makes GPTQ significantly more accurate than naive round-to-nearest.

GPTQ typically operates at 4-bit (GPTQ-4bit) or 3-bit. Available via `auto-gptq` library.

### 6. AWQ: Activation-Aware Weight Quantization

AWQ (Lin et al., 2023) observes that not all weights are equally important. The weights corresponding to high-activation-magnitude input channels are more sensitive to quantization error. AWQ scales these salient weights by a factor before quantization (making them larger, thus more accurately representable with the same number of bits), and compensates by scaling activations inversely.

AWQ is faster to apply than GPTQ (no Hessian computation) and achieves similar or better accuracy. Available via `autoawq` library.

### 7. Double Quantization (QLoRA)

QLoRA introduces "double quantization": quantize the quantization constants themselves. In standard 4-bit quantization, each group of 64 weights shares a float32 scale factor. In double quantization, these scale factors are themselves quantized to 8-bit. This saves an additional ~0.37 bits per parameter on average.

Double quantization is implemented in bitsandbytes via `bnb_4bit_use_double_quant=True`. You will use this in Week 33.

### 8. GGUF and llama.cpp

GGUF is a file format (used by `llama.cpp`) that enables CPU-based inference with quantized models. For deployment on Mac (Apple Silicon) without a GPU, GGUF with Q4_K_M or Q5_K_M quantization is the practical standard. Not used in training, but important for deployment. You will encounter this in Phase 5–6.

---

## Practical Comparison

| Format | Bits | Memory (7B model) | Quality loss | Use case |
|---|---|---|---|---|
| FP32 | 32 | 28 GB | None | Training baseline |
| BF16 | 16 | 14 GB | Minimal | Standard training/inference |
| FP16 | 16 | 14 GB | Minimal | Inference, mixed-precision training |
| INT8 (LLM.int8) | 8 | 7 GB | Small | Memory-constrained inference |
| GPTQ-4bit | 4 | 3.5 GB | Small-Medium | Efficient inference |
| AWQ-4bit | 4 | 3.5 GB | Small-Medium | Efficient inference |
| NF4 (bitsandbytes) | 4 | 3.5 GB | Small | QLoRA training |

---

## Connections

**Builds on:** Week 30–31's LoRA math — quantization is the second half of QLoRA.

**Needed for:** Week 33 (QLoRA = NF4 base model + LoRA adapters). Every inference decision from Phase 5 onward.

---

## Common Misconceptions / Pitfalls

- **"INT4 and NF4 are the same."** No — NF4 uses non-uniform quantile-based levels optimized for normal distributions; INT4 uses evenly spaced levels. NF4 is significantly better for LLM weights.
- **"GPTQ quantization can be done without a GPU."** GPTQ requires a GPU for the Hessian computation. AWQ is faster but also benefits from GPU.
- **"Quantized models cannot be fine-tuned."** This was true before QLoRA. QLoRA adds trainable LoRA adapters on top of a frozen quantized model.
- **"8-bit quantization is always better than 4-bit."** Not always — GPTQ-4bit with good calibration can outperform naïve INT8 quantization.

---

## Time Allocation (6–8 hrs)

| Activity | Time |
|---|---|
| Read LLM.int8() paper abstract + key sections | 1h |
| Read GPTQ and AWQ paper abstracts + sections 3–4 | 1.5h |
| Read Maarten Grootendorst's visual guide to quantization | 30m |
| Set up bitsandbytes, quantize Qwen2.5-Coder-1.5B to 4-bit | 1.5h |
| Measure model size, VRAM, inference speed; build comparison table | 1.5h |
| Commit results to GitHub | 30m |
