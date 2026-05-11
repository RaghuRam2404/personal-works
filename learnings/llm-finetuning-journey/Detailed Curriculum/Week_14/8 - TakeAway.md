# Week 14 TakeAway — LLaMA Papers and Production Code

**This week in 15 words:** LLaMA is your Week 12 nanoGPT, engineered at scale with RMSNorm, SwiGLU, RoPE, and GQA.

---

## LLaMA Architecture Quick Reference

```
# LLaMA 1 / 2 / 3 core config (8B variant)
d_model (hidden_size)       = 4096
n_layers                    = 32
n_heads (num_attention_heads) = 32
n_kv_heads (num_key_value_heads) = 8 (LLaMA 3), 32 (LLaMA 1/2 7B)
d_k (head_dim)              = 128
intermediate_size           = 14336  (LLaMA 3 8B SwiGLU hidden)
vocab_size                  = 128256 (LLaMA 3); 32000 (LLaMA 1/2)
rope_theta                  = 500000 (LLaMA 3); 10000 (LLaMA 1/2)
context_length              = 8192 (LLaMA 3 base)
weight_tying                = False  (unlike GPT-2)
```

---

## `modeling_llama.py` Class Map

| Class | What it does | Week 12 equivalent |
|---|---|---|
| `LlamaRMSNorm` | RMSNorm | `RMSNorm` |
| `LlamaRotaryEmbedding` | Precomputes RoPE cos/sin | `precompute_rope_freqs` |
| `apply_rotary_pos_emb` | Applies rotation to Q, K | `apply_rotary_emb` |
| `LlamaMLP` | SwiGLU FFN | `SwiGLUMLP` |
| `repeat_kv` | Expands n_kv_heads → n_heads | `repeat_interleave` |
| `LlamaAttention` | GQA with KV cache | `CausalSelfAttention` (GQA) |
| `LlamaDecoderLayer` | One transformer block | `Block` |
| `LlamaForCausalLM` | Full model + LM head | `GPT` |

---

## Key Differences LLaMA vs. GPT-2

| Property | GPT-2 | LLaMA 3 8B |
|---|---|---|
| Weight tying | Yes | No |
| Positional encoding | Learned (wpe) | RoPE (theta=500000) |
| Normalization | LayerNorm, Post | RMSNorm, Pre |
| FFN | GELU, 4x dim | SwiGLU, 14336 |
| Attention | MHA | GQA (n_kv=8) |
| Vocab size | 50257 | 128256 |

---

## Numbers to Remember

- LLaMA 3 8B: `n_heads=32, n_kv=8, d_k=128, d_model=4096, vocab=128256, rope_theta=500000`
- GQA savings vs MHA: `n_kv / n_heads` fraction of KV cache (e.g., 8/32 = 25% of MHA cost)
- `repeat_kv n_rep = n_heads / n_kv_heads = 4` for LLaMA 3 8B
- Over-training beyond Chinchilla optimum → better inference efficiency

---

## Red Flags When Loading LLaMA

- Wrong `rope_theta` in config → silent degradation at long contexts.
- `num_key_value_heads` not set → defaults to MHA (full KV heads), OOM risk.
- Loading LLaMA 2-Chat when you want base → safety filters will reject SQL-style queries during fine-tuning prep.
- Grad checkpointing off with batch > 1 at 7B → OOM on any single 24GB GPU.
