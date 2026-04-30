# Week 14 Resources

## Papers

- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) — Touvron et al. 2023. The original LLaMA paper. Read fully this week. Take handwritten notes on Section 2 (approach) and Section 3 (architecture).
- [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288) — Touvron et al. 2023. Focus on Section 2 (pretraining) and the architecture table.
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) — Meta AI 2024. Read Sections 1, 3 (pre-training), and 5 (results). Section 3 covers data, architecture, and training recipe in detail.

## Videos

- [Umar Jamil — LLaMA explained: KV-Cache, Rotary Positional Embedding, RMS Norm, Grouped Query Attention, SwiGLU](https://www.youtube.com/watch?v=Mn_9W1nCFLo) — Umar Jamil — 1h40m. Revisit this video this week alongside `modeling_llama.py`. Maps directly to the HF code.

## Blog Posts / Articles

- [LLaMA 2: Full Breakdown](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/) — Meta AI official page. Good starting point before the paper.
- [Understanding LLaMA — Architecture Deep Dive](https://blog.briankitano.com/llama-from-scratch/) — Walks through implementing LLaMA from scratch in PyTorch; good companion to your annotation task.

## GitHub Repos

- [HuggingFace modeling_llama.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — The primary code to annotate this week.
- [meta-llama/llama](https://github.com/meta-llama/llama) — Original Meta implementation. Simpler than HF's but less production-complete. Good for cross-referencing.
- [meta-llama/llama3](https://github.com/meta-llama/llama3) — LLaMA 3 original code. Compare the `model.py` to LLaMA 1.

## Documentation

- [HuggingFace LLaMA Configuration](https://huggingface.co/docs/transformers/model_doc/llama#transformers.LlamaConfig) — All config parameters for the HF LLaMA model. Study `num_attention_heads`, `num_key_value_heads`, `rope_theta`, `intermediate_size`.
- [meta-llama/Meta-Llama-3-8B config.json](https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/config.json) — The actual config file for LLaMA 3 8B. Every parameter, exactly as used.

## Optional / Bonus

- [Alpaca: A Strong, Replicable Instruction-Following Model](https://crfm.stanford.edu/2023/03/13/alpaca.html) — Stanford CRFM blog. Describes how LLaMA 1 was instruction-tuned with 52k GPT-generated examples in $600. Direct predecessor to Phase 4 of your curriculum.
- [Mistral 7B](https://arxiv.org/abs/2310.06825) — Mistral AI 2023. Extends LLaMA architecture with Sliding Window Attention and GQA. Good comparison point once you've read LLaMA.
