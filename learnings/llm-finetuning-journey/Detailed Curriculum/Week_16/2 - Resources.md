# Week 16 Resources

This week is a review and gate week — no new reading is required, but the standard sections below consolidate the most useful papers, videos, and repos from Phase 2 (Weeks 9–15) for gate preparation.

## Videos

- [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Andrej Karpathy — ~1h57m. Rewatch any specific section if causal mask, weight tying, or the training loop is unclear.
- [Coding LLaMA 2 from scratch in PyTorch](https://www.youtube.com/watch?v=Mn_9W1nCFLo) — Umar Jamil — ~2h59m. Rewatch the RoPE, RMSNorm, GQA, and SwiGLU sections if any modern-architecture component is shaky.
- [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU) — Andrej Karpathy — ~4h01m. Rewatch the gradient accumulation and mixed-precision sections if the training loop from Week 15 still feels fragile.

## Papers

- [Attention Is All You Need (Vaswani et al. 2017)](https://arxiv.org/abs/1706.03762) — Re-read Sections 3.1–3.3 if multi-head attention or sinusoidal positional encoding is shaky.
- [Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau et al. 2014)](https://arxiv.org/abs/1409.0473) — Re-read Section 3 if the additive attention score function is unclear.
- [RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE)](https://arxiv.org/abs/2104.09864) — Section 3.4 (pseudocode) if the RoPE implementation is unclear.
- [GQA: Training Generalized Multi-Query Transformer Models (Ainslie et al. 2023)](https://arxiv.org/abs/2305.13245) — Section 3 (KV-cache memory analysis).
- [The Curious Case of Neural Text Degeneration (Holtzman et al. 2020, Nucleus Sampling)](https://arxiv.org/abs/1904.09751) — Section 2 (top-p algorithm).
- [LLaMA: Open and Efficient Foundation Language Models (Touvron et al. 2023)](https://arxiv.org/abs/2302.13971) — Tables 1 and 2 (architecture configs).
- [The Llama 3 Herd of Models (Llama Team 2024)](https://arxiv.org/abs/2407.21783) — Table 3 (8B config: n_heads, n_kv, rope_theta).

## GitHub Repos

- [karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt) — Reference `train_gpt2.py` for the final training-loop structure from Week 15.
- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) — The minimal GPT codebase you re-implemented and modified across Phase 2.
- [huggingface/transformers — modeling_llama.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — Your annotated reference from Week 14; compare against your own implementation during gate review.

## Blog Posts / Articles

- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) — Trace through the Transformer code once more if any implementation detail still feels unclear.
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) — Best visual reference for the KV cache and decoder-only attention flow.

## Documentation

- [PyTorch nn.MultiheadAttention](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) — Reference for the Phase 2 attention API; useful when comparing your hand-rolled implementation against the library version.
- [PyTorch Mixed Precision (torch.amp)](https://pytorch.org/docs/stable/amp.html) — Reference for the autocast/GradScaler patterns from Week 15.

## Optional / Bonus

- [Karpathy nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero) — All Karpathy notebooks from micrograd through GPT; the `gpt-from-scratch.ipynb` mirrors Weeks 11–15.
- [Sebastian Raschka — Understanding Encoder And Decoder LLMs](https://magazine.sebastianraschka.com/p/understanding-encoder-and-decoder) — Concise comparison of encoder-only, decoder-only, and encoder-decoder transformers; useful framing before Phase 3 pretraining.

## Phase 3 Preview

When you pass Phase 2, Phase 3 covers:
- Scaling laws (Chinchilla, Kaplan et al.) — Week 17
- Pretraining data composition — Week 18
- Training your own 50M-parameter model from scratch — Weeks 19–24

Everything in Phase 3 builds directly on Phase 2. The math will be new; the transformer code is what you built this phase.
