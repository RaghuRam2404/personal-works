# Week 13 Resources

## Papers

- [The Curious Case of Neural Text Degeneration (Nucleus Sampling)](https://arxiv.org/abs/1904.09751) — Holtzman et al. 2019. The top-p (nucleus) sampling paper. Read Sections 1–3. Table 1 shows the failure modes of beam search and top-k.
- [Fast Transformer Decoding: One Write-Head is All You Need (MQA)](https://arxiv.org/abs/1911.02150) — Shazeer 2019. Original multi-query attention paper that introduced the concept behind GQA.
- [Speculative Decoding](https://arxiv.org/abs/2211.17192) — Leviathan et al. 2022. Optional — explains how small draft models speed up large model inference.

## Videos

- [Umar Jamil — LLaMA 2 explained](https://www.youtube.com/watch?v=Mn_9W1nCFLo) — Umar Jamil — 1h40m. Revisit specifically the KV cache section (timestamp ~25m). Best visual explanation of how the cache grows during generation.

## Blog Posts / Articles

- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) — Jay Alammar. The KV cache animation is the clearest visualization available. Study the "Self-attention at decoding time" section.
- [How to generate text: using different decoding methods for language generation with Transformers](https://huggingface.co/blog/how-to-generate) — HuggingFace blog. Compares greedy, beam search, top-k, top-p with examples. Code uses HF generate(), but the concepts apply directly.
- [Transformers KV Caching Explained](https://medium.com/@joaolages/kv-caching-explained-276520203249) — João Lages. Good from-scratch explanation with diagrams.

## GitHub Repos

- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) — Reference `generate()` in `model.py`. Compare with your KV-cached version.
- [vllm-project/vllm](https://github.com/vllm-project/vllm) — Production-grade LLM serving with PagedAttention KV cache management. Preview of Phase 5+ inference tooling.

## Documentation

- [HuggingFace Generation Strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies) — Complete documentation of all sampling strategies available in `model.generate()`. Read the "Decoding strategies" section.
- [torch.multinomial](https://pytorch.org/docs/stable/generated/torch.multinomial.html) — The PyTorch function for sampling from a probability distribution. Understand the `replacement` parameter.

## Optional / Bonus

- [PagedAttention (vLLM)](https://arxiv.org/abs/2309.06180) — Kwon et al. 2023. The memory management system behind vLLM that treats KV cache like OS virtual memory. Production-grade KV cache management.
- [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05100) — Pope et al. 2022. Google's analysis of LLM inference arithmetic. Useful for understanding the compute vs. memory bandwidth tradeoff.
