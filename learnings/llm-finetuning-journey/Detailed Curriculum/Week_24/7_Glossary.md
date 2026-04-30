# Week 24 Glossary — SOTA Pretraining Recipes

**GQA (Grouped Query Attention)**: A modification to multi-head attention where multiple query heads share a single key-value head, reducing the KV cache size during inference.

**MoE (Mixture of Experts)**: A transformer variant where FFN layers are replaced with multiple "expert" networks; a router selects a subset of experts per token, enabling large total parameter counts with low active compute.

**MLA (Multi-Head Latent Attention)**: DeepSeek's technique for compressing the KV cache by projecting keys and values into a low-dimensional latent space before storing.

**FIM (Fill-in-the-Middle)**: A training objective for code models where the model predicts a missing middle section given the surrounding prefix and suffix; improves code completion.

**SwiGLU**: An activation function combining Swish and Gated Linear Unit, used in FFN layers of most modern LLMs; outperforms GELU and ReLU on language modeling.

**RoPE (Rotary Position Embeddings)**: A relative positional encoding scheme that enables length extrapolation; used in Llama 3 and Qwen models.

**GQA KV heads**: The number of Key-Value head groups in GQA; Llama 3-8B uses 8 KV heads and 32 Query heads.

**FP8 training**: Training using 8-bit floating-point format for activations and gradients; reduces memory and bandwidth requirements by 2× compared to BF16.

**DPO (Direct Preference Optimization)**: A post-training alignment method that fine-tunes the model to prefer human-preferred outputs over rejected ones without a separate reward model.

**GRPO (Group Relative Policy Optimization)**: An RLHF variant used by Qwen that optimizes policy relative to a group of sampled responses; more stable than PPO for reasoning tasks.

**Tiktoken**: OpenAI's BPE tokenizer implementation; used (or variants of it) in Llama 3, Qwen2.5, and DeepSeek models.

**Hypertable**: A TimescaleDB concept for partitioned time-series tables; generated automatically from a PostgreSQL table using `create_hypertable()`.

**time_bucket()**: A TimescaleDB function for grouping time-series data into fixed time intervals; the TimescaleDB equivalent of date_trunc() with custom intervals.

**Continuous aggregates**: Precomputed materialized views in TimescaleDB that are automatically refreshed; essential for efficient time-series queries.

**Post-training**: The phase after base pretraining where the model is fine-tuned for instruction following, safety, and specialized capabilities via SFT, DPO, RLHF, or GRPO.
