# Week 16 Resources

This week has no new reading. All resources are reviews of prior weeks.

---

## Phase 2 Review Resources (by topic)

### Attention Mechanism (Weeks 9–10)
- [Bahdanau et al. 2014](https://arxiv.org/abs/1409.0473) — re-read Section 3 if shaky on score function
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — re-read Sections 3.1–3.3 if shaky on MHA or positional encoding
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) — trace through the code once more if implementation feels unclear

### Decoder-Only / nanoGPT (Week 11)
- [Karpathy nanoGPT video](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Andrej Karpathy — ~2h25m. rewatch specific sections if causal mask or weight tying is unclear

### Modern Architecture (Week 12)
- [RoPE paper](https://arxiv.org/abs/2104.09864) — Section 3.4 (pseudocode) if RoPE implementation is unclear
- [GQA paper](https://arxiv.org/abs/2305.13245) — Section 3 (KV cache memory analysis)
- [Umar Jamil LLaMA video](https://www.youtube.com/watch?v=Mn_9W1nCFLo) — Umar Jamil — ~2h15m. rewatch RMSNorm and SwiGLU sections

### KV Cache and Sampling (Week 13)
- [The Illustrated GPT-2 (KV cache section)](https://jalammar.github.io/illustrated-gpt2/) — best visual reference
- [Nucleus Sampling paper](https://arxiv.org/abs/1904.09751) — Section 2 (top-p algorithm)

### LLaMA (Week 14)
- [LLaMA 1 paper](https://arxiv.org/abs/2302.13971) — Tables 1 and 2 (architecture configs)
- [LLaMA 3 paper](https://arxiv.org/abs/2407.21783) — Table 3 (8B config: n_heads, n_kv, rope_theta)
- [modeling_llama.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — your annotated version from Week 14

### GPT-2 Reproduction (Week 15)
- [Karpathy GPT-2 video](https://www.youtube.com/watch?v=l8pRSuU81PU) — Andrej Karpathy — ~4h. rewatch gradient accumulation and mixed precision sections if shaky
- [build-nanogpt repo](https://github.com/karpathy/build-nanogpt) — reference `train_gpt2.py` for the final loop structure

---

## Phase 3 Preview

When you pass Phase 2, Phase 3 covers:
- Scaling laws (Chinchilla, Kaplan et al.) — Week 17
- Pretraining data composition — Week 18
- Training your own 50M-parameter model from scratch — Weeks 19–24

Everything in Phase 3 builds directly on Phase 2. The math will be new; the transformer code is what you built this phase.
