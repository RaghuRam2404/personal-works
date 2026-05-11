# Week 64 Quiz — Quantization Part 2

## Multiple Choice

**Q1.** You run GPTQ quantization on your 7B SQL model with `damp_percent=0.01` and observe that several layers produce `NaN` weights after quantization. What is the most likely cause and fix?

A. The calibration dataset is too small; increase to 2,048 examples
B. The Hessian inversion is numerically unstable; increase `damp_percent` to 0.1 or 0.2
C. The model has BF16 weights which GPTQ does not support; convert to FP32 first
D. `desc_act=False` is incompatible with Qwen2.5; set it to True

**Q2.** Your AWQ-quantized SQL model has 2.1 percentage points lower accuracy than the BF16 baseline on your 200-example benchmark. Your colleague suggests this is because you used WikiText-103 for calibration instead of SQL examples. Which of the following best explains why calibration domain matters for AWQ?

A. AWQ uses calibration data to compute perplexity; SQL sentences have lower perplexity than Wikipedia
B. AWQ identifies salient channels by observing which channels have large activation magnitudes on the calibration data; SQL inputs activate different channels than natural language, so the wrong channels are protected
C. AWQ rounding is deterministic and calibration data only affects quantization speed, not quality
D. The tokenizer behaves differently on SQL vs. natural language, causing embedding layer misalignment

**Q3.** You want to deploy your model on a customer's laptop with 8 GB RAM and no GPU. Which format is most appropriate?

A. AWQ INT4, because it has the best throughput
B. GPTQ INT4, because it is the most accurate
C. Q4_K_M GGUF with llama.cpp CPU inference
D. BF16 with 8-bit bitsandbytes, because it preserves most accuracy

**Q4.** You compare Q4_K_M GGUF (4.5 GB) and AWQ INT4 (4.2 GB) on the same GPU. AWQ achieves 85 tok/s while GGUF achieves 38 tok/s. The most important factor driving this gap is:

A. GGUF compresses weights more aggressively, causing slower dequantization
B. AWQ uses GEMM kernels optimized for GPU tensor cores; GGUF dequantizes on CPU in the llama.cpp default configuration
C. AWQ stores weights in FP16 internally and only compresses metadata
D. GGUF lacks support for batched inference, making it inherently slower

## Short Answer

**Q5.** Explain why you quantize a merged model rather than a LoRA adapter checkpoint. What would happen if you attempted to quantize an unmerged adapter?

**Q6.** Your GPTQ perplexity on a held-out SQL test set is 9.8, but the AWQ perplexity is 9.1 despite both being INT4. Propose two hypotheses for why GPTQ is worse here, and describe an experiment to test each.

**Q7.** You are preparing model cards for three Hub repos. A user unfamiliar with your project wants to reproduce your SQL accuracy numbers. List the five pieces of information your model card must include to make reproduction possible.

## Deep Scenario

**Q8.** Your team lead asks you to recommend a single quantized format for production. You have three requirements:

- Must run on a cloud GPU instance with 8 GB VRAM (A10G)
- Must achieve at least 80 tok/s for a real-time SQL assistant
- Accuracy must be within 2 percentage points of the BF16 baseline on your 200-example benchmark

Your measurements show: AWQ INT4 hits 85 tok/s, 0.3 pp accuracy drop, 5.0 GB VRAM. GPTQ INT4 hits 72 tok/s, 1.1 pp accuracy drop, 5.2 GB VRAM. Q4_K_M GGUF hits 38 tok/s on CPU, N/A GPU.

Write a recommendation memo (200–300 words) that: (a) selects a format and justifies it against all three requirements, (b) identifies the one remaining risk with your chosen format, and (c) proposes a monitoring strategy for detecting accuracy degradation in production.
