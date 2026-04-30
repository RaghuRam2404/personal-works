# Week 63 Resources — Quantization Formats: GGUF, GPTQ, AWQ

## Papers

[GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) — Frantar et al. 2022; the Hessian-based layer-wise quantization algorithm that set the INT4 standard.

[AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) — Lin et al. 2023; explains salient-channel protection and per-channel scaling strategy.

[LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339) — Dettmers et al. 2022; foundational outlier-aware quantization paper; important context for understanding why naive INT8 fails at 7B+ scale.

[QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks](https://arxiv.org/abs/2402.04396) — Tseng et al. 2024; pushes 2-bit quantization to near-lossless accuracy; reference for where the field is heading.

## Videos

[Quantization Explained — Andrej Karpathy / Tim Dettmers commentary (Yannic Kilcher)](https://www.youtube.com/watch?v=mii-xFm11fQ) — ~45 min; covers INT8/INT4 intuition, outlier problem, and LLM.int8 walkthrough.

[llama.cpp deep dive — Georgi Gerganov at AI Engineer Summit](https://www.youtube.com/watch?v=s9_oSqVsOVk) — ~30 min; the author explains GGUF design decisions and K-quant naming.

## Blog Posts / Articles

[Tim Dettmers — A Gentle Introduction to 8-bit Matrix Multiplication for Transformers](https://huggingface.co/blog/hf-bitsandbytes-integration) — Hugging Face blog; excellent conceptual walkthrough with memory math.

[The Illustrated Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization) — Maarten Grootendorst; visual guide covering absmax, zero-point, GPTQ, and GGUF with diagrams.

[Unsloth GGUF Export Guide](https://docs.unsloth.ai/basics/export-to-gguf) — Official Unsloth documentation; covers the exact `save_pretrained_gguf` API and quantization type options used in your workflow.

[Hugging Face Quantization Concepts](https://huggingface.co/docs/transformers/main/en/quantization/overview) — Official HF docs overview; links to bitsandbytes, GPTQ, AWQ, and GGUF integration guides.

## GitHub Repos

[ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) — The reference GGUF inference engine; `quantize` binary and `convert_hf_to_gguf.py` are the tools you use this week.

[casper-hansen/AutoAWQ](https://github.com/casper-hansen/AutoAWQ) — Production AWQ quantization library; supports Qwen2.5 out of the box.

[PanQiWei/AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) — The standard GPTQ library for HuggingFace models; handles calibration data and layer-wise quantization.

[unslothai/unsloth](https://github.com/unslothai/unsloth) — The training framework you used; also provides `model.save_pretrained_gguf()` with direct quantization level control.

## Documentation

[Hugging Face GPTQ Integration](https://huggingface.co/docs/transformers/main/en/quantization/gptq) — Complete API reference for `GPTQConfig` and `AutoModelForCausalLM.from_pretrained` with quantization.

[Hugging Face AWQ Integration](https://huggingface.co/docs/transformers/main/en/quantization/awq) — API docs for `AwqConfig` and loading AWQ-quantized models.

[llama.cpp GGUF quantization types reference](https://github.com/ggerganov/llama.cpp/blob/master/docs/quantization.md) — Official table of all Q-types, their bit depth, and perplexity benchmarks on Llama models.

## Optional / Bonus

[QuaRot: Outlier-Free 4-bit Inference in Rotated LLMs](https://arxiv.org/abs/2404.00456) — Recent technique using random rotation to eliminate outliers before quantization; good reading on where GPTQ/AWQ limitations come from.

[GGUF format specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) — Low-level binary format spec; useful if you want to understand what is actually stored in the `.gguf` file header.

[Exllamav2](https://github.com/turboderp/exllamav2) — GPU-optimized GPTQ inference kernel faster than standard AutoGPTQ; worth knowing for deployment.
