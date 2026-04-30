# Week 32 Resources

## Papers

- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339) — Dettmers et al. 2022. The LLM.int8() paper; read sections 1–3 on outlier features.
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) — Frantar et al. 2022. The GPTQ paper; read sections 2–3 for the Hessian-based compensation method.
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) — Lin et al. 2023. Read sections 1–3; pay attention to the salience scoring approach.
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Dettmers et al. 2023. Section 2.2 introduces NF4 and double quantization.

## Videos

No required videos this week. Use time for benchmarking experiments.

## Blog Posts / Articles

- [A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization) — Maarten Grootendorst. Excellent visual explainer of all major formats. Required reading.
- [Making LLMs Even More Accessible with bitsandbytes, 4-bit Quantization and QLoRA](https://huggingface.co/blog/4bit-transformers-bitsandbytes) — HuggingFace blog. Practical guide to using bitsandbytes.

## GitHub Repos

- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) — Tim Dettmers. The library for NF4/INT8 quantization used in QLoRA. Read the README carefully.
- [auto-gptq](https://github.com/AutoGPTQ/AutoGPTQ) — Community GPTQ implementation for HuggingFace models.
- [autoawq](https://github.com/casper-hansen/AutoAWQ) — AWQ implementation for HuggingFace models.

## Documentation

- [HuggingFace Quantization Guide](https://huggingface.co/docs/transformers/main/en/quantization/overview) — Overview of all quantization methods supported in transformers.
- [bitsandbytes integration docs](https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes) — How to use `BitsAndBytesConfig` with transformers.

## Optional / Bonus

- [SqueezeLLM: Sparse-Quantization for Efficient LLM Inference](https://arxiv.org/abs/2306.07629) — Alternative approach using sparse + dense decomposition; context for understanding the quantization trade-off space.
- [QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks](https://arxiv.org/abs/2402.04396) — 2024 state-of-the-art in 2-bit quantization; shows how far the field has advanced.
