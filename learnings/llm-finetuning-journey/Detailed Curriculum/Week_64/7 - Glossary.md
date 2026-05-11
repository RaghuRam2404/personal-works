# Week 64 Glossary — Quantize and Publish Your Final Model

**Merged model**: A HuggingFace model directory where LoRA adapter weights have been added into the base weight matrices; the only valid input to quantization tools.

**F16 GGUF**: An intermediate GGUF file storing full float16 weights; the source file for all subsequent llama.cpp quantization steps.

**`llama-quantize`**: The llama.cpp binary that reads an F16 GGUF and writes a lower-bit GGUF at the specified quantization level (e.g., Q4_K_M).

**`llama-perplexity`**: The llama.cpp binary that computes per-token perplexity on a text file using a GGUF model; used to measure quantization quality without running full benchmarks.

**GEMM version (AWQ)**: The AWQ kernel variant that uses matrix-matrix multiply on GPU tensor cores; required for batched or throughput-sensitive inference on Ampere+ GPUs.

**`damp_percent`**: GPTQ hyperparameter that adds a diagonal term to the Hessian before inversion to prevent numerical instability; typical value 0.05–0.2.

**`desc_act`**: GPTQ option that reorders weights by activation magnitude before quantization, improving quality at the cost of non-contiguous memory access during inference.

**Model card**: The `README.md` file in a HuggingFace Hub repository that documents model provenance, training details, evaluation results, and usage instructions.

**`upload_file`**: HuggingFace Hub API function for pushing a single binary file (e.g., a GGUF) to a repository, bypassing the git-lfs model directory conventions.

**Perplexity degradation budget**: The tolerable increase in perplexity from quantization; typically set at less than 10% relative to BF16 baseline (e.g., BF16 PPL 8.4 → quantized PPL < 9.2).
