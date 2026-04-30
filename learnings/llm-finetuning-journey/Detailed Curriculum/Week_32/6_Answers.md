# Week 32 Quiz Answers

## Q1 — Answer: B

**Answer:** B. BF16 has the same 8-bit exponent as FP32, preventing overflow.

**Why:** FP16 uses only 5 exponent bits, giving a maximum representable value of 65504. In LLM training, gradients can easily exceed this — producing Inf or NaN, which crashes training. BF16 uses 8 exponent bits (same as FP32), so its maximum value is comparable to FP32 (~3.4 × 10^38). The cost is fewer mantissa bits (7 vs. 23 in FP32), meaning lower precision in representing weight values — but for training stability, range matters more than precision.

---

## Q2 — Answer: B

**Answer:** B. Outlier activations exceeding threshold 6.0 trigger FP16 path.

**Why:** Dettmers' LLM.int8() paper showed that large language models (>6B parameters) develop systematic "outlier features" — specific activation dimensions that consistently take values 100x or more larger than typical. These outliers dominate quantization error if quantized with INT8. LLM.int8() detects these outlier columns per-batch, extracts them into a separate FP16 multiply, and computes the remaining "normal" columns in INT8. This mixed decomposition preserves near-FP16 quality while achieving ~2x memory reduction.

---

## Q3 — Answer: C

**Answer:** C. All three fit: BF16 (14GB), INT8 (7GB), NF4 (3.5GB).

**Why:** A 7B model in BF16 = 7×10^9 × 2 bytes = 14GB — fits in 16GB with some overhead. INT8 = 7GB — fits comfortably. NF4 (with grouping overhead, actual storage ~4–4.5GB for 7B) — fits easily. All three are viable on 16GB VRAM for inference; training (which adds gradients + optimizer states) is a different question. Only NF4 with LoRA (QLoRA) makes training feasible on 16GB.

---

## Q4 — Answer: B

**Answer:** B. Normally distributed weights + quantile-based levels.

**Why:** Most LLM weight tensors, when plotted as histograms, show a bell-shaped distribution centered near zero. INT4 places its 16 levels uniformly spaced, wasting precision in the sparse tails and coarsely representing the dense center. NF4 places its 16 values at the 1/16, 2/16, ..., 16/16 quantiles of a standard normal distribution. This maximizes the expected number of weights that fall exactly on a quantization level, minimizing expected quantization error for this distribution. Hence: information-theoretically optimal given the observed distribution.

---

## Q5 — Answer: C

**Answer:** C. AWQ is faster — no inverse Hessian computation required.

**Why:** GPTQ's per-layer Hessian computation can take 5–30 minutes per model depending on size and calibration set. AWQ avoids this by using a different insight: it identifies salient weight channels based on activation magnitude statistics (which are cheap to compute — just a forward pass through calibration data) and protects those channels from quantization error by scaling. AWQ runs in minutes rather than tens of minutes.

**Why B is wrong:** AWQ does use a calibration dataset (a small set of examples for computing activation statistics), though a smaller one than GPTQ.

---

## Q6 — Short Answer

The concern is throughput: INT8 matrix multiplications on current GPU hardware (specifically NVIDIA GPUs before Hopper H100) are not always faster than FP16. On Ampere A100 and earlier, INT8 operations use the Tensor Core's INT8 path which theoretically doubles FLOPS compared to FP16. However, the LLM.int8() implementation (bitsandbytes) processes outlier dimensions in FP16, adding kernel overhead and reducing the practical speedup. In practice, LLM.int8() may be the same speed or slightly slower than BF16 while using half the memory. On H100 with FP8 support, the situation improves. The memory saving (7GB vs. 14GB) is real and allows batching larger inputs — which may offset the per-token latency cost through higher throughput overall.

---

## Q7 — Short Answer

Double quantization (from QLoRA) quantizes the quantization constants themselves. In standard 4-bit quantization, each group of 64 weights shares a single FP32 scale factor (4 bytes, adding 4/64 = 0.0625 bits per parameter overhead). Double quantization takes these FP32 scale factors and quantizes them to 8-bit using a second quantization step, reducing the scale factor overhead to roughly 8/64/n bytes (where n is the second-level group size). The resulting memory saving is approximately 0.37 bits per parameter. The quality trade-off is minimal because scale factors are smooth and vary slowly, making them easy to represent at 8-bit precision.

---

## Q8 — Short Answer

Decision framework for accepting quantization quality loss:

1. **Compute relative degradation:** (40% − 37%) / 40% = 7.5% relative drop. Is 7.5% acceptable for the task?
2. **Check against minimum viable threshold:** If your production requirement is >35% exact match, NF4 is fine. If your requirement is >40%, it fails.
3. **Test execution correctness, not just exact match:** Exact match underestimates semantic correctness — two different SQL queries can return the same rows. Run both BF16 and NF4 models on execution correctness (Week 39's eval harness) before deciding.
4. **Consider the QLoRA path:** You will fine-tune the NF4 model with LoRA in Week 33, which will recover the 3% gap and likely improve beyond the BF16 base. The base model quality drop is not the final metric.

Recommendation: accept the 3% drop on the base quantized model, fine-tune with QLoRA, and measure fine-tuned quality before deciding.

---

## Q9 — Scenario Answer

**Options to evaluate:**

1. BF16 (baseline, single-query serving)
2. INT8 LLM.int8() (7GB, potential for batch size 2–3)
3. GPTQ-4bit (3.5GB, potential for batch size 4–6)
4. AWQ-4bit (3.5GB, same as GPTQ, faster to deploy)

**Experiment:**
- Load each quantized version
- Measure: execution correctness on held-out 100 SQL examples (must stay >85%)
- Measure: latency at batch size 1 and batch size 4 (p95 latency)
- On A100 40GB: BF16 allows batch 2; GPTQ-4bit allows batch 8+

**Recommendation:** AWQ-4bit or GPTQ-4bit. At 500 queries/day (average ~0.35 req/min), latency SLA is not a peak-load concern — but 4-bit quantization enables larger batch sizes if load spikes. If the quality evaluation shows >85% execution correctness at 4-bit, prefer GPTQ-4bit for better inference kernel optimization on A100. Deploy with a 10-example quality spot-check during production monitoring.
