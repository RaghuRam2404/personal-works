# Week 63 Answers

## Q1 — Answer: B

**Why:** At batch size 1 (single-user inference), each auto-regressive step requires loading the full set of model weights to compute one output token. For a 7B model at bf16 (14GB), this creates a minimum "read time" determined by memory bandwidth, regardless of compute speed. Modern GPUs are so fast at matrix multiplication that their compute is vastly underutilized during single-token generation — the limiting factor is how fast you can get the weights from HBM into the compute units. Quantization to INT4 directly addresses this bottleneck by reducing weight size.

---

## Q2 — Answer: B

**Why:** The A100's memory bandwidth (900–2,000 GB/s depending on generation) vs. Apple M2's unified memory bandwidth (~68–100 GB/s) is the primary speed driver. At batch size 1, inference is memory-bandwidth-bound. The A100 can load weights roughly 13× faster than M2 unified memory. Despite quantization making the model smaller (which helps M2), the raw bandwidth advantage of A100 dominates for large batch scenarios. Note: for batch size 1, the Mac's throughput (18 tok/s) may actually be comparable to or exceed A100 on a per-device-cost basis, which is the argument for local deployment.

---

## Q3 — Answer: C

**Why:** AWQ's key insight (from their activation analysis) is that a small fraction of weight channels (< 1%) correspond to input activations that are consistently very large in magnitude. When you quantize these weights with standard grid quantization, the relative error is high (the weight might round significantly relative to its true value). Because large activations amplify small weight errors, these are the quantization errors that most degrade model output. AWQ protects them by scaling them up before quantization (effectively giving them more of the quantization range) and scaling the corresponding activations back down, preserving output magnitude.

---

## Q4 — Answer: B

**Why:** GPTQ uses calibration data to estimate the second-order curvature (Hessian) of the loss with respect to each weight, then uses this to weight the quantization error. Weights that have high curvature (where small changes cause large loss changes) are quantized more carefully. If calibration data is general English text (WikiText-2), the Hessian estimates reflect importance for next-word prediction on English text — not SQL generation. Weights critical for TimescaleDB function syntax may be estimated as low-importance and quantized aggressively, leading to accuracy loss on your target domain.

---

## Q5 — Model Answer

A 2pp drop (65% → 63%) is generally acceptable for most deployment scenarios. It represents a 3% relative degradation — within the noise floor of evaluation variation.

Context-dependence:
- **Local personal use:** 2pp is fully acceptable. The benefit (running on a Mac without GPU) far outweighs the 2pp accuracy cost.
- **Developer tool / CLI:** Acceptable. Users expect occasional SQL errors and can re-run.
- **Production API for a business application:** Depends on the cost of failure. If a wrong SQL query deletes data, 2pp matters. If wrong queries just need to be retried, it is acceptable.
- **Benchmark comparison vs. GPT-4o:** Here, 2pp matters. Report both bf16 and Q4_K_M scores separately in your technical report.

The answer: acceptable for deployment, must be disclosed in technical documentation.

---

## Q6 — Model Answer

Recommend: Q4_K_M GGUF via Ollama.

Reason: GGUF is the only format designed for CPU-only inference without a GPU. It includes optimized CPU kernels (AVX2/AVX-512 on x86, NEON on ARM) that make inference practical without GPU acceleration. The Q4_K_M variant at ~4.4GB fits comfortably in 8GB RAM (leaving headroom for OS and the KV cache).

Practical user experience: on a modern Intel i7 or M-series Mac with 8GB RAM:
- First token latency: 3–8 seconds (model loading + first token) 
- Throughput: 3–8 tok/s on x86 CPU, 12–18 tok/s on Apple Silicon M-series
- A typical SQL query (150 tokens) takes 20–50 seconds to generate on x86, 8–12 seconds on Apple Silicon

This is usable for offline/batch use but too slow for real-time interactive use on x86. On Apple Silicon, it is interactive enough. Recommend Ollama for the user experience (automatic model management, REST API, simple CLI).

---

## Q7 — Model Answer

Diagnosis: TimescaleDB-specific queries involve rare tokens (`time_bucket_gapfill`, `locf`, `interpolate`, `compress_chunk`, etc.) that appear in a small fraction of the training data. In the model's weight matrices, the directions corresponding to these rare tokens are associated with weights that have unusual magnitudes (outliers relative to the common SQL patterns). K-quants quantizes each block independently, but a block containing both common SQL weights and TimescaleDB-specific weights will have the quantization scale dominated by the common weights. The rare TimescaleDB-specific weights in that block suffer higher relative quantization error.

Targeted fix without switching formats:

1. **Use Q5_K_M instead of Q4_K_M for TimescaleDB-specific layers.** GGUF allows layer-specific quantization via custom quantization strategies. Identify the layers most responsible for TimescaleDB token generation (typically the final MLP layers and specific attention heads) and quantize only those layers at Q5_K_M while keeping other layers at Q4_K_M. llama.cpp supports per-layer quantization via its `--custom-q` flag. This increases model size by ~400MB but may recover 8–12pp on TimescaleDB accuracy.

2. **Fine-tune the quantized model.** After Q4_K_M quantization, run a brief QLoRA fine-tuning pass (100–200 steps) on your TimescaleDB-specific training examples. This is "quantization-aware adaptation" — you are adapting the LoRA parameters on top of the quantized base to recover the TimescaleDB accuracy loss. The LoRA additions are in bf16 and thus high-precision, compensating for the quantized base model's accuracy loss on rare patterns.
