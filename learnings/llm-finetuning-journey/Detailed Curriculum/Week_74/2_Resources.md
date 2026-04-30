# Week 74 Resources — Context Extension: LongRoPE and YaRN

## Papers (primary reading this week)

[YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) — Peng et al. 2023; NTK-aware scaling and temperature adjustment; focus on Sections 3 (method) and 5 (results).

[LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753) — Ding et al. 2024; evolutionary search for per-dimension scaling; focus on Sections 3 (method) and 5 (ablations comparing to YaRN).

[RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — Su et al. 2021; the original RoPE paper; read Sections 1–3 if RoPE is unfamiliar.

## Supporting Papers

[Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) — Chen et al. 2023; the naive position interpolation baseline that YaRN and LongRoPE improve on.

[FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Dao 2023; the memory-efficient attention implementation required for long-context inference.

[LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding](https://arxiv.org/abs/2308.14508) — Bai et al. 2023; the benchmark used to evaluate long-context models; useful for measuring your model's long-context SQL ability.

## Videos

[YaRN and Long Context Explained (Trelis Research)](https://www.youtube.com/watch?v=4J8rjvnkl1Y) — ~25 min; visual explanation of RoPE, position interpolation, and YaRN's NTK-aware approach.

[Flash Attention 2 Deep Dive (Tri Dao, Stanford)](https://www.youtube.com/watch?v=zy8ChVd_oTM) — ~40 min; memory analysis and implementation details.

## Blog Posts / Articles

[Extending Context Length with YaRN (Nous Research blog)](https://huggingface.co/blog/yarn) — Practical YaRN implementation guide with code examples for HuggingFace models.

[How Long Can Open-Source LLMs Truly Focus? (Shi et al.)](https://arxiv.org/abs/2307.03172) — Tests whether long-context models actually use the full context; relevant for evaluating whether context extension improves SQL on large schemas.

## GitHub Repos

[jquesnelle/yarn](https://github.com/jquesnelle/yarn) — The original YaRN implementation and fine-tuning scripts; includes the HuggingFace config format.

[microsoft/LongRoPE](https://github.com/microsoft/LongRoPE) — Official LongRoPE code from Microsoft Research; includes evolutionary search implementation.

## Documentation

[HuggingFace — rope_scaling configuration](https://huggingface.co/docs/transformers/model_doc/llama#transformers.LlamaConfig.rope_scaling) — Config format for YaRN and other rope_scaling types in HuggingFace transformers; same format used for Qwen2.5.

## Optional / Bonus

[LM-Infinite: Zero-Shot Extreme Length Generalization for Large Language Models](https://arxiv.org/abs/2308.16137) — A training-free approach to long context; compares favorably to fine-tuning-based methods at moderate extensions.

[Schema-Linking for Text-to-SQL (Guo et al.)](https://arxiv.org/abs/1911.06399) — The algorithmic basis for schema compression; automatically identifies relevant tables for a query using entity linking.
