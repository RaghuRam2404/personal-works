# Week 14 Assignment Solutions

This week has no coding tasks. The "solutions" are reference answers to help you verify your reading comprehension.

---

## Task 1 Reference Answers

**Q1 — Training data:** LLaMA 1 uses: CommonCrawl 67%, C4 15%, GitHub 4.5%, Wikipedia 4.5%, Gutenberg+Books3 4.5%, ArXiv 2.5%, StackExchange 2%. GitHub code is included because natural language and code are related — code training improves structured reasoning, logical inference, and later helps with SQL generation. Code also contains high-quality technical English.

**Q2 — Chinchilla insight:** Chinchilla (Hoffmann et al. 2022) showed that the compute-optimal number of training tokens for a model of size N is approximately 20N. LLaMA 7B was trained on 1T tokens (vs. 7B × 20 = 140B compute-optimal tokens). By training much longer, the model extracts more learning per parameter. A 13B model with 1T tokens has more effective knowledge than a 175B model with 300B tokens (the GPT-3 training budget).

**Q3 — Pre-RMSNorm:** The paper states they adopt Pre-Norm for training stability, following Wang et al. (2019). Pre-RMSNorm means applying RMSNorm before the attention and FFN sublayers, not after. This ensures the residual stream receives unnormalized values from the sublayer output, making gradient flow more predictable and enabling training larger models without instability.

**Q4 — SwiGLU dimension:** LLaMA uses `2/3 × 4 × d_model = 8/3 × d_model`, rounded to the nearest multiple of 256. The rounding ensures tensor dimensions are multiples of hardware-friendly sizes (CUDA GEMM efficiency). For d_model=4096: `int(8/3 × 4096) = 10923` → rounded to 11008. This is slightly different from a naive `8/3` calculation.

**Q5 — Weight tying:** LLaMA 1 does NOT tie the embedding and LM head weights (unlike GPT-2). The LM head is a separate `nn.Linear(d_model, vocab_size, bias=False)`. The paper does not give an explicit justification — in practice at large scale, the parameter savings from tying are proportionally small (32k × 4096 / 7B ≈ 2%), so the simplicity of separate weights was preferred.

---

## Task 2 Reference Answers

**Q1 — LLaMA 1 → LLaMA 2 changes:**
- Context length: 2048 → 4096
- Training tokens: 1T → 2T
- GQA: introduced for 34B and 70B models (n_kv_heads=8); 7B/13B still use full MHA
- Ghost Attention: enables better multi-turn instruction following in Chat variants

**Q2 — Vocabulary effect:** A 128k vocabulary tokenizes text into ~30% fewer tokens than a 32k vocabulary on typical English text. Fewer tokens → shorter sequences → fewer forward passes for the same amount of text → lower training cost for a given number of training examples. It also improves multilingual tokenization efficiency significantly.

**Q3 — RoPE theta:** Larger theta means slower decay of the rotation frequencies with position. At theta=10000, position embeddings for positions > 2048 start to repeat or interfere. At theta=500000, the frequencies are so low that positional information stays distinct across 100k+ positions. This allows the model to distinguish positions in very long contexts without the sinusoidal ambiguity that limited LLaMA 1/2.

**Q4 — GQA memory savings:**
- Full MHA: `2 × 32 × 128 × 4096 × 32 × 2 bytes = 2.1 GB`
- GQA n_kv=8: `2 × 8 × 128 × 4096 × 32 × 2 bytes = 0.54 GB`
- Savings: 4x less KV cache memory (from 32 to 8 KV heads).

---

## Task 3 — Key Annotation Notes

**`repeat_kv`:**
```python
def repeat_kv(hidden_states, n_rep):
    # hidden_states: [batch, num_kv_heads, seq, head_dim]
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, seq, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, seq, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq, head_dim)
```
This is equivalent to `repeat_interleave(n_rep, dim=1)` but uses `expand` (zero-copy broadcasting) + `reshape` for efficiency. Called after computing K and V, before the attention dot product.

**Key difference from your Week 12 code:** HF uses `expand + reshape`; Week 12 used `repeat_interleave`. Both are correct; `expand + reshape` avoids an explicit data copy.

---

## How to Verify Your Reading Comprehension

1. Close the papers. Answer: "What are the 4 architectural changes in LLaMA vs. original Transformer?" Answer should include: RMSNorm, SwiGLU, RoPE, Pre-norm. Bonus: no weight tying.
2. Open `modeling_llama.py`. Point to where GQA happens without searching. You should point to `repeat_kv` and the `num_key_value_groups` attribute.
3. State from memory: LLaMA 3 8B — vocab_size, rope_theta, n_kv_heads, context length. Answers: 128256, 500000, 8, 8192.
