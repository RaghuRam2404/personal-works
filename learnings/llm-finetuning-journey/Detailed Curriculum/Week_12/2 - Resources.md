# Week 12 Resources

## Papers

- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) — Xiong et al. 2020. Pre-LN vs. Post-LN analysis. Read Figure 1 and Section 3.
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) — Zhang & Sennrich 2019. The RMSNorm paper. Short (6 pages). Read fully.
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — Noam Shazeer 2020. SwiGLU paper. 4 pages. Read fully — Table 1 shows all GLU variants.
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — Su et al. 2021. The RoPE paper. Read Sections 1–3 and the pseudocode in Section 3.4.
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) — Ainslie et al. 2023. Short paper. Read Section 3 on the KV cache memory analysis.
- [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) — Beltagy et al. 2020. Original Sliding Window Attention paper. Optional context.

## Videos

- [Umar Jamil — LLaMA explained: KV-Cache, Rotary Positional Embedding, RMS Norm, Grouped Query Attention, SwiGLU](https://www.youtube.com/watch?v=Mn_9W1nCFLo) — ~1h10m. Excellent explanation of all four improvements with code walkthrough. Watch before coding.

## Blog Posts / Articles

- [Rotary Embeddings: A Relative Revolution](https://blog.eleuther.ai/rotary-embeddings/) — EleutherAI blog. Detailed explanation with math and diagrams. Read alongside the paper.
- [Understanding GQA and MQA](https://huggingface.co/blog/kv-cache-quantization) — HuggingFace blog. Explains KV cache and how GQA reduces it, with diagrams.

## GitHub Repos

- [meta-llama/llama](https://github.com/meta-llama/llama) — Original LLaMA implementation. Look at `model.py` — all four improvements are clearly implemented.
- [huggingface/transformers — modeling_llama.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — The production HuggingFace LLaMA implementation. Read the `LlamaRMSNorm`, `LlamaRotaryEmbedding`, `LlamaMLP`, and `LlamaAttention` classes.

## Documentation

- [PyTorch F.silu](https://pytorch.org/docs/stable/generated/torch.nn.functional.silu.html) — The SiLU/Swish activation used in SwiGLU.
- [torch.nn.functional.scaled_dot_product_attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) — PyTorch 2.0+ Flash Attention-compatible attention function. Stretch goal this week.

## Optional / Bonus

- [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) — RoPE extension technique that allows LLaMA to handle 128k+ context. Preview of context scaling techniques.
- [Flash Attention 2](https://arxiv.org/abs/2307.08691) — Dao et al. 2023. The algorithm behind `F.scaled_dot_product_attention`. Understanding it is worth your time before Week 15.
