# Week 14 — Reading and Understanding the LLaMA Papers

## Learning Objectives

By end of this week, you will be able to:

- Describe the key design decisions in LLaMA 1 and explain why each was made
- Compare the architectural and training differences across LLaMA 1, 2, and 3
- Read and annotate `modeling_llama.py` from the HuggingFace transformers library
- Explain `num_key_value_heads` in the HF implementation and how it implements GQA
- Write a 1-page summary mapping `modeling_llama.py` to your Week 12 nanoGPT modernization
- Identify where RoPE, RMSNorm, SwiGLU, and GQA appear in production code

## Time Allocation (6–8 hrs)

| Activity | Time |
|---|---|
| Read LLaMA 1 paper fully (2302.13971) | 2 hrs |
| Read LLaMA 2 paper — arch changes section (2307.09288) | 0.75 hrs |
| Read LLaMA 3 paper — Sections 1, 3, 5 (2407.21783) | 1 hr |
| Print and annotate `modeling_llama.py` line by line | 2.5 hrs |
| Write `journal.md` summary | 0.75 hrs |

This is a reading-heavy week. There is no new coding. The coding deliverable is annotation and written notes.

---

## Concepts

### Why LLaMA Matters

LLaMA (Large Language Model Meta AI), released by Meta AI in February 2023, was the first high-quality open-weights LLM family. Before LLaMA, the dominant models were GPT-3, PaLM, Chinchilla — all closed weights, API-only. LLaMA changed the landscape: researchers could now fine-tune, quantize, and modify a competitive model on consumer hardware.

LLaMA 1 (7B, 13B, 30B, 65B) was trained on only open-source data. The 7B variant runs on a single GPU. This is the model that spawned Alpaca, Vicuna, WizardLM, and ultimately made open-source LLM fine-tuning a mainstream activity — including everything you'll do in Phases 4–6 of this curriculum.

### LLaMA 1 — Key Design Decisions

**Architecture:** Decoder-only transformer (like GPT), with:
- Pre-RMSNorm (RMSNorm before each sublayer, not after)
- SwiGLU activation (with 2/3 × 4 × d_model intermediate dim, rounded to 256 multiple)
- RoPE instead of learned positional embeddings
- No tied embeddings (unlike GPT-2) — the LM head is separate

**Training data:** 1.4 trillion tokens from public data: CommonCrawl (67%), GitHub (4.5%), Wikipedia (4.5%), Project Gutenberg (4.5%), StackExchange (2%), ArXiv (2.5%), books (4.5%).

**Key insight:** Touvron et al. showed that a smaller model trained on more data outperforms a larger model trained on less. This is the Chinchilla principle (see Week 17) applied in practice. LLaMA 7B was trained on 1T tokens — far more than the compute-optimal amount for a 7B model at the time. The result: LLaMA 13B outperforms GPT-3 175B on most benchmarks.

**No instruction tuning in LLaMA 1:** LLaMA 1 is a base model only. It cannot follow instructions. The fine-tuned variants (Alpaca, Vicuna) were trained separately.

### LLaMA 2 — Key Changes

LLaMA 2 (2023, Touvron et al.) introduced several important changes:

- **Context length:** 4096 tokens (vs. 2048 in LLaMA 1)
- **Grouped-Query Attention (GQA):** For the 34B and 70B models, `n_kv_heads = 8` vs. `n_heads = 64`. The 7B model still uses standard MHA. This is where GQA moved from paper to production.
- **Ghost Attention:** A technique for multi-turn instruction following in the Chat variants.
- **Instruction-tuned variants (LLaMA 2-Chat):** Trained with RLHF (reinforcement learning from human feedback). This is the version most users interact with. For the base model, LLaMA 2 is architecturally very similar to LLaMA 1.

### LLaMA 3 — Key Changes

LLaMA 3 (2024, "Llama 3: A Herd of Models") is a significant upgrade:

- **Tokenizer:** New 128k vocabulary BPE tokenizer (vs. 32k in LLaMA 1/2). Larger vocabulary → fewer tokens per document → more efficient training.
- **GQA for all sizes:** Even the 8B model now uses `n_q_heads=32, n_kv_heads=8` (GQA with group size 4).
- **Training data:** 15+ trillion tokens. More data, better data quality.
- **Context length:** 8192 tokens during pretraining (extended to 128k via long-context fine-tuning in LLaMA 3.1).
- **RoPE theta:** 500,000 (vs. 10,000 in LLaMA 1/2). Larger theta → more gradual frequency decay → better long-context generalization.

### Reading `modeling_llama.py`

The HuggingFace implementation at [modeling_llama.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) is approximately 1000 lines. This week you will read and annotate every significant class and function.

Key classes to annotate:

**`LlamaRMSNorm`:** Your `RMSNorm` from Week 12, exactly. Note: it uses `hidden_states.to(torch.float32)` for numerical stability before computing the norm, then casts back.

**`LlamaRotaryEmbedding`:** Precomputes cos/sin tables. The `LlamaLinearScalingRotaryEmbedding` and `LlamaDynamicNTKScalingRotaryEmbedding` variants handle context extension — out of scope for now, but note they exist.

**`LlamaMLP`:** Your SwiGLU. Note the naming: `gate_proj`, `up_proj`, `down_proj` — exactly the same names you used in Week 12.

**`LlamaAttention`:** The full GQA attention. Key fields:
- `num_heads`: query heads (e.g., 32)
- `num_key_value_heads`: KV heads (e.g., 8)
- `num_key_value_groups = num_heads // num_key_value_heads` (= 4)
- The `repeat_kv` function expands KV heads to match query heads before attention

**`LlamaDecoderLayer`:** One block: self-attention + MLP, with Pre-RMSNorm on each.

**`LlamaModel`:** The full stack of N `LlamaDecoderLayer`s.

**`LlamaForCausalLM`:** Adds the LM head on top of `LlamaModel`. Note: LLaMA does NOT tie LM head weights to embeddings (unlike GPT-2).

### Differences Between Your Week 12 nanoGPT and `modeling_llama.py`

| | Your Week 12 nanoGPT | LLaMA (HF) |
|---|---|---|
| Norm | RMSNorm ✓ | RMSNorm ✓ |
| FFN | SwiGLU ✓ | SwiGLU ✓ |
| Position | RoPE ✓ | RoPE ✓ |
| Attention | MHA or GQA (yours) | GQA with `repeat_kv` |
| Weight tying | Yes (GPT-style) | No |
| LM head bias | No | No |
| Norm position | Pre-LN ✓ | Pre-RMSNorm ✓ |
| Context length | 128–256 | 2048–8192 |

You have already implemented the core of LLaMA. The production version is a more careful engineering of the same ideas.

## Common Misconceptions / Pitfalls

- **"LLaMA 1 used instruction tuning."** No — LLaMA 1 is a base model. Alpaca was a fine-tuned version trained separately on GPT-4-generated instruction data.
- **Confusing LLaMA 2-Chat and LLaMA 2 base.** They have the same architecture but very different behavior. The Chat model is RLHF-tuned; it adds instruction-following and safety alignment. For fine-tuning your SQL model in Phase 4, you'll start from the base, not Chat.
- **`num_key_value_heads` in HF config.** If this is None or equal to `num_attention_heads`, GQA is not active (full MHA). In LLaMA-3 8B config, `num_attention_heads=32, num_key_value_heads=8`.
- **`repeat_kv` is not learning.** It just broadcasts the 8 KV heads to 32 query heads by repeating — no parameters involved.
- **LLaMA 3's RoPE theta change.** LLaMA 3 uses `rope_theta=500000`. If you load a LLaMA 3 checkpoint but initialize RoPE with theta=10000, the positional embeddings will be completely wrong and generation will be garbage.

## Connections

This week builds on the full Phase 2 architecture stack: Week 10 (Transformer fundamentals), Week 11 (decoder-only GPT family), and especially Week 12 (modernized nanoGPT with RMSNorm, SwiGLU, RoPE, and GQA). Everything you implemented in Week 12 appears in `modeling_llama.py` — the purpose of this reading week is to close the gap between your toy implementation and production code. You cannot meaningfully annotate this file without having written the components yourself.

Week 15 depends on this reading: the GPT-2 124M reproduction requires you to understand the differences between GPT-style weight tying and LLaMA-style separate LM heads, and to recognize which implementation patterns come from which design choice. Beyond the immediate curriculum, all of Phase 4 (fine-tuning, Weeks 28–40) and Phase 5 (alignment, Weeks 41–52) operate on Llama-style models. If you skip or skim this week's annotation work, you will encounter `modeling_llama.py` lines during fine-tuning debugging that you will not understand.
