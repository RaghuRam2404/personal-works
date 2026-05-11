# Week 74 — Frontier Reading 4: Context Extension — LongRoPE and YaRN

## Learning Objectives

By the end of this week, you will be able to:

- Explain why RoPE (Rotary Position Embedding) limits context length and why naive context extension fails
- Describe how YaRN extends context via Non-Uniform interpolation and NTK-aware scaling
- Describe how LongRoPE uses evolutionary search to find non-uniform rescaling factors
- Assess whether context extension is relevant to your PostgreSQL SQL use case
- Identify when longer context would and would not improve your model's SQL accuracy

## Concepts

### Why Context Length Matters for SQL

Your current model uses a 4096-token context window. A typical SQL prompt contains:
- System prompt: ~50 tokens
- Database schema: 200–1500 tokens (depends on table count and column count)
- Question: 20–80 tokens
- Generated SQL: 50–300 tokens

For most queries, 4096 tokens is sufficient. But consider these failure cases:
- A database with 20+ tables and 100+ columns: schema alone can exceed 3000 tokens
- Multi-turn conversations (Week 76): accumulate conversation history in context
- Analytical queries with CTEs: the reasoning context may span thousands of tokens

Context extension techniques allow you to serve these larger schemas without truncating critical information.

### RoPE and Why Naive Extension Fails

RoPE (Su et al. 2021) encodes token position by rotating the query and key vectors in attention by an angle proportional to position:

Q_m = R_m Q,   K_n = R_n K

where R_m is a rotation matrix at position m. The inner product Q_m · K_n depends on the relative position (m - n) through the rotation: it is a function of the position difference, not absolute positions. This gives RoPE its key property: models trained with RoPE generalize across positions up to the training context length.

The failure at extension: during training, the model only sees position rotations corresponding to positions 0 to T (e.g., T=4096). At positions T+1 and beyond, the rotation angles are out-of-distribution — the model has never seen these rotations during training. Directly extending to longer context causes catastrophic accuracy degradation.

Naive position interpolation scales down all positions by a factor λ = new_context / old_context, so position 8192 maps to position 4096 in the old scale. This prevents out-of-distribution rotations but causes a new problem: the high-frequency components of the RoPE encoding are compressed, losing fine-grained position distinctions. Short-range attention (attending to nearby tokens) degrades because nearby positions now have similar rotation angles.

### YaRN: Non-Uniform Interpolation and NTK-Aware Scaling

YaRN (Peng et al. 2023, arXiv 2309.00071) solves the naive interpolation problem with two techniques:

NTK-aware scaling: Instead of scaling all position dimensions uniformly, YaRN applies different scaling factors to different frequency components of the RoPE embedding. High-frequency components (which encode short-range position distinctions) are not interpolated — they keep their original high-frequency values. Low-frequency components (which encode long-range position distinctions) are scaled to accommodate the extended context.

The intuition: think of RoPE like a clock with two hands — a "second hand" (high frequency, fast rotation) and an "hour hand" (low frequency, slow rotation). To extend context, you want to slow down the hour hand (extend the long-range range) without affecting the second hand (preserve short-range resolution).

Temperature scaling (attention entropy): YaRN also adjusts the attention softmax temperature for extended contexts. At very long contexts, attention becomes spread across too many positions, causing the model to lose focus. YaRN scales down the temperature to maintain attention sharpness.

YaRN fine-tuning: the model is fine-tuned for a small number of steps (400–1000 steps) at the extended context length on long documents, allowing it to adapt its attention patterns to the new range.

Result: YaRN extends Llama 2 7B from 4096 to 128K tokens with perplexity degradation of < 0.5 points at 64K context, requiring only 400 fine-tuning steps.

### LongRoPE: Non-Uniform Dimension Rescaling

LongRoPE (Ding et al. 2024, arXiv 2402.13753) takes a different approach to the same problem. Instead of a analytically designed scaling function (like YaRN's NTK formula), LongRoPE uses evolutionary search to find the optimal rescaling factor for each RoPE dimension independently.

The key insight: different RoPE frequency dimensions (there are d/2 of them, where d is the head dimension) may need different rescaling factors. YaRN applies a formula that is analytically motivated but not optimal. LongRoPE searches for the optimal per-dimension factor.

The evolutionary search algorithm:
1. Start with a population of rescaling factor vectors, each of length d/2
2. Evaluate each candidate by measuring perplexity on a long-context validation set (after a small fine-tuning run)
3. Apply mutation and crossover operations to generate the next population
4. After N generations, select the best-performing rescaling vector

The result: LongRoPE achieves lower perplexity at very long contexts (128K–2M tokens) than YaRN with the same fine-tuning budget, because the per-dimension optimization outperforms the analytically derived NTK formula.

LongRoPE also introduces a two-stage approach: extend to a very long context (e.g., 128K) with the searched factors, then fine-tune briefly at 128K. This avoids the perplexity degradation at short contexts that naive long-context training causes.

### Practical Application to Your SQL Model

Should you apply context extension to postgres-sqlcoder-7b?

Cases where 4096 tokens is insufficient:
- Schemas with 15+ tables and 100+ columns: these can exceed 3000 tokens, leaving only 1000 tokens for question + SQL
- Multi-turn SQL workflows: 5 turns of question + SQL ≈ 5 × 400 = 2000 tokens of history

Cases where context extension would help:
- Run YaRN fine-tuning to extend to 8192 tokens: cost is 400 steps ≈ 0.5 GPU-hours
- Use the 8192-context model for schemas that currently get truncated

Quick YaRN implementation for Qwen2.5 (which already uses RoPE):
```python
# In HuggingFace transformers, enable YaRN by patching rope_scaling
config = AutoConfig.from_pretrained("./postgres-sqlcoder-7b-final")
config.rope_scaling = {
    "type": "yarn",
    "factor": 2.0,           # extend to 2x context (4096 → 8192)
    "original_max_position_embeddings": 4096,
}
model = AutoModelForCausalLM.from_pretrained("./postgres-sqlcoder-7b-final", config=config)
# Fine-tune for 400 steps on long-schema examples
```

## Connections

Context extension is directly applicable to Week 76 (multi-turn agentic SQL) where conversation history will quickly fill a 4096-token context. YaRN's 400-step fine-tuning cost is low enough to apply in Week 75 or 76 as an extension. The techniques in this week are also relevant to reading any LLM paper that mentions context length — knowing YaRN and LongRoPE lets you evaluate whether a paper's long-context claims are methodologically sound.

## Common Misconceptions / Pitfalls

Context extension does not give you "free" longer context. The model must be fine-tuned at the extended length to adapt its attention patterns. Without fine-tuning, even with correct RoPE scaling, the model's attention may degrade at the extended range.

LongRoPE's evolutionary search is expensive (requires many fine-tuning runs). For practical extension to 2x or 4x your training context, YaRN is faster and nearly as good.

Longer context increases inference cost quadratically with standard attention (O(n²)). At 8192 tokens, memory and compute are 4x higher than at 4096 tokens. Use Flash Attention 2 to make this practical.

## Time Allocation (6–8 hours)

- 1.5h: Read YaRN paper (arXiv 2309.00071) — focus on Method section (Sections 3–4)
- 1.5h: Read LongRoPE paper (arXiv 2402.13753) — focus on Method section and ablations
- 1.0h: Read "RoPE" original paper (arXiv 2104.09864) if RoPE is unfamiliar — Sections 1–3
- 1.5h: Implement YaRN rope_scaling config for Qwen2.5; test perplexity on a 6000-token SQL schema prompt
- 0.5h: Write synthesis notes in `reading_notes/week74_synthesis.md`
