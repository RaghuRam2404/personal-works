# Week 63 — Quantization Deep Dive Part 1: GGUF, GPTQ, AWQ Comparison

## Learning Objectives

By the end of this week, you will be able to:

- Explain the mathematical basis of post-training quantization (PTQ) for LLMs
- Describe the key differences between GGUF/llama.cpp, GPTQ, and AWQ quantization methods
- Choose the appropriate quantization format for each deployment scenario
- Understand the accuracy-throughput-memory trade-off at each quantization level
- Read a quantization comparison table and interpret what the numbers mean for your use case

## Why Quantization Matters

Your final model is 14GB at bf16. Almost no one can run a 14GB model on consumer hardware. Quantization makes it accessible:
- Q4_K_M GGUF: ~4.5GB → runs on 8GB RAM Mac, Raspberry Pi, or any laptop
- AWQ INT4: ~4GB → runs on RTX 4060 (8GB VRAM) for real-time inference
- GPTQ INT4: ~4GB → runs on any CUDA GPU with 8GB+ VRAM

Beyond accessibility, quantized models are faster: fewer bits to load from memory means higher throughput. A Q4 model can generate tokens 2–3× faster than bf16 on the same hardware because memory bandwidth, not compute, is the bottleneck for 7B models on modern GPUs.

## Concepts

### The Memory Bandwidth Bottleneck

For transformer inference (not training), the bottleneck is almost never FLOPS — it is the bandwidth required to load model weights from memory into the compute units for each token generation step. At 7B parameters × 2 bytes (bf16) = 14GB, each auto-regressive step requires loading up to 14GB of weights. At typical GPU memory bandwidth (900 GB/s for H100, 300 GB/s for A100, 68 GB/s for M1 Mac), this determines the speed ceiling.

Quantization to INT4 (0.5 bytes/parameter): 7B × 0.5 = 3.5GB weights. Loading takes proportionally less time. For a small batch size (inference, not training), the effective throughput increases by approximately the ratio of precision reduction: 2× for INT8, 3.5× for INT4.

### GGUF Format and llama.cpp

**GGUF** (GPT-Generated Unified Format) is the file format for llama.cpp quantized models. It is:
- A single binary file containing weights + tokenizer + model config
- Designed for CPU inference (though can use GPU offloading)
- Supports multiple quantization levels: Q8_0 (8-bit), Q6_K (6-bit, grouped), Q5_K_M, Q4_K_M (4-bit medium), Q4_K_S (4-bit small), Q3_K_M, Q2_K

**Q4_K_M** is the "sweet spot" for most use cases:
- Size: ~4.5GB for a 7B model
- Quality: within 2–5% of bf16 on most benchmarks
- Speed: 15–35 tokens/second on Apple Silicon Mac
- "K" = K-quants (uses different precision for different weight matrices)
- "M" = medium quality (better than "S"mall, cheaper than "L"arge)

**GGUF quantization method:** GGUF uses a combination of abs-max quantization and k-means clustering of weight blocks. The "K-quants" algorithm groups blocks of weights (typically 256 or 64) and quantizes each block independently, choosing the quantization scale that minimizes the maximum absolute error within that block.

### GPTQ (Generalized Post-Training Quantization)

GPTQ (Frantar et al., 2022) uses the Optimal Brain Compression (OBC) framework to minimize the L2 reconstruction error of each weight after quantization. The key insight: rather than quantizing each weight independently, GPTQ updates the remaining (not yet quantized) weights to compensate for the quantization error introduced in the already-quantized weights.

**GPTQ process:**
1. Load calibration data (small dataset, typically 128 examples)
2. For each layer, quantize weights one-by-one using the inverse Hessian of the loss to find the optimal quantization value
3. After quantizing each weight, update remaining weights to minimize reconstruction error

**GPTQ format:** Stored as INT4 or INT8 per weight, with FP16 scales and zeros. Compatible with `auto-gptq` library and many inference backends.

**Advantages:** High accuracy at INT4 — typically within 1–3% of bf16 on most benchmarks. Well-supported in the HuggingFace ecosystem.

**Disadvantages:** Requires calibration data and calibration runtime (20–60 minutes for a 7B model). CPU-unfriendly (designed for GPU inference). Less portable than GGUF.

### AWQ (Activation-aware Weight Quantization)

AWQ (Lin et al., 2023) identifies and protects the weights that matter most for model accuracy by analyzing activation magnitudes. The observation: a small fraction of weights (< 1%) correspond to large activation values and have disproportionate impact on model quality. Quantizing these weights aggressively causes most of the accuracy degradation.

**AWQ process:**
1. Run calibration data through the model; record activation magnitudes per channel
2. Identify "salient" channels where activations are consistently large
3. Scale these channels up before quantization (so they map to a finer quantization grid) and scale them down after (equivalent to higher precision for those weights without changing the number of bits stored)
4. Quantize all weights to INT4

**AWQ advantages:** No activation outlier problem (GPTQ struggles with models that have activation outliers); works well with INT4 for group sizes of 64 or 128; compatible with vLLM for high-throughput serving.

**AWQ disadvantages:** Also requires calibration data. Less portable than GGUF.

### Comparing the Three Methods

| Method | Target hardware | Format | Quality at INT4 | Speed | Notes |
|--------|----------------|--------|-----------------|-------|-------|
| GGUF (Q4_K_M) | CPU + MPS (Mac) | Single file | Within 3–5% of bf16 | 15–35 tok/s on Mac | Best for local deployment |
| GPTQ INT4 | CUDA GPU | HuggingFace shards | Within 1–3% of bf16 | 60–120 tok/s on A100 | GPU-only; requires auto-gptq |
| AWQ INT4 | CUDA GPU | HuggingFace shards | Within 1–2% of bf16 | 80–150 tok/s on A100 | Best quality at INT4; vLLM native |

### What Does "Within X% of bf16" Actually Mean?

This shorthand hides important nuances. A model that is "within 3%" overall may be within 1% on simple SQL and within 10% on complex TimescaleDB hyperfunctions. Quantization errors are not uniformly distributed — they concentrate in weights that were previously at unusual magnitudes (outliers).

For your SQL model, the most likely degradation pattern: TimescaleDB-specific SQL (rare patterns) may be more affected by quantization than common PostgreSQL patterns (frequent patterns → weights are more "canonical" and quantize better). This is an empirical claim — you verify it in Week 64.

### Common Misconceptions and Pitfalls

**"Quantization is lossless at Q4."** False — Q4 is lossy. The accuracy degradation depends on the model, the task, and the quantization method. Always measure before deploying.

**"AWQ is always better than GPTQ."** AWQ has better theoretical properties for activation outlier handling, but GPTQ has a larger ecosystem and is sometimes more accurate in practice for specific models. Benchmark both.

**"K-quants in GGUF are the same as group quantization in GPTQ."** Similar concept (independent quantization per block) but different implementations and scale choices. Not directly comparable.

## Time Allocation (6–8 hrs)

- 2h: Read and take notes on GPTQ paper abstract + Section 2–3 (methodology)
- 2h: Read AWQ paper abstract + Section 3 (methodology)
- 1h: Study the GGUF K-quants documentation in the llama.cpp repository
- 2h: Create the comparison study: run a reference 7B model (e.g., the base Qwen) through all three quantization methods on 100 BIRD-SQL questions; compare accuracy and speed
- 1h: Document findings in `quantization_comparison_study.md`

## Connections

This week builds on Week 32 (quantization fundamentals: INT8, INT4, dynamic vs. static quantization, and the accuracy-memory tradeoff), which gave you the theoretical vocabulary for this deeper dive. Week 32 covered what quantization is; Week 63 covers how the three dominant production formats — GGUF, GPTQ, and AWQ — implement it differently and when to choose each. The Phase 4 and Phase 6 fine-tuning runs are also prerequisites: your intuition about which weight patterns matter for SQL query generation will inform how you interpret quantization sensitivity results.

Week 64 applies the knowledge from this week to quantize your actual final model. The comparison study you build this week (running all three methods on a reference 7B model) is practice for the Week 64 work and gives you baseline expectations for accuracy degradation. Weeks 65–66 then deploy the quantized variants: GGUF for local Ollama deployment, and either GPTQ or AWQ for the cloud API endpoint. If Week 63's study reveals a clear winner for your query distribution, you can skip the alternative in Week 64 and go straight to the preferred format.
