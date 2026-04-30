# Week 74 Assignment Solutions

## Task 1: YaRN Key Formulas

NTK-aware scaling in 3 sentences: The Neural Tangent Kernel (NTK) perspective on position embeddings suggests that interpolating all dimensions uniformly changes the effective bandwidth of high-frequency features — like compressing a high-resolution image loses fine details. "NTK-aware" means the scaling is applied non-uniformly: high-frequency RoPE dimensions (which distinguish nearby positions) are not interpolated, while low-frequency dimensions (which distinguish far positions) are scaled to accommodate the extended range. This preserves short-range attention quality (the model can still tell token 5 from token 6) while extending long-range range (the model can attend to positions beyond 4096).

YaRN attention temperature: YaRN scales the attention logits by a factor `t` that decreases with context length. The formula is approximately `t = 0.1 * ln(s) + 1` where `s` is the context scale factor. For s=2 (doubling context), t ≈ 1.07. This slightly sharpens attention at longer contexts, compensating for the natural spread of attention weights over more tokens.

Fine-tuning requirement from the paper: approximately 400 steps at the extended context length, with 10% of training data being long examples and 90% short examples (to maintain short-context performance).

## Task 2: LongRoPE Evolutionary Search

The evolutionary search in LongRoPE: the search space is a vector of d/2 rescaling factors (one per RoPE frequency dimension, typically 64 factors for a 128-dim head). The objective function is perplexity on a held-out long-context validation set after a short fine-tuning run. The algorithm runs N=100–200 generations with population size 64. Each generation evaluates each candidate (50+ GPU-minutes per candidate in the paper's setup), then applies mutation (random perturbation of factors) and crossover (combining two candidate vectors).

Practical budget assessment: LongRoPE's evolutionary search requires hundreds of short fine-tuning runs, costing 50–200 GPU-hours total. For extending to 2x context (4096→8192), YaRN is dramatically more practical: 0.5 GPU-hours vs 100+ GPU-hours. LongRoPE becomes worthwhile only for very large extensions (128K+) where the per-dimension optimization provides a meaningful quality difference.

Two-stage LongRoPE: Stage 1 trains at the very long target context (e.g., 128K) with the searched rescaling factors. Stage 2 re-tunes at a shorter intermediate context (e.g., 16K or 32K) to prevent short-context perplexity degradation that Stage 1 introduces. The two-stage approach allows the model to "have both" — strong long-context performance and preserved short-context quality.

## Task 3: YaRN Config for Qwen2.5

```python
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

config = AutoConfig.from_pretrained("./postgres-sqlcoder-7b-final")

# Apply YaRN scaling for 2x context extension (4096 → 8192)
config.rope_scaling = {
    "type": "yarn",
    "factor": 2.0,
    "original_max_position_embeddings": 4096,
}
# Note: Qwen2.5 already has a 128K context natively via its own RoPE scaling.
# For Qwen2.5, you may not need YaRN — check if the base model supports
# longer context out of the box via its "dynamic" rope_scaling type.

model = AutoModelForCausalLM.from_pretrained(
    "./postgres-sqlcoder-7b-final",
    config=config,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2"  # required for long contexts
)
tokenizer = AutoTokenizer.from_pretrained("./postgres-sqlcoder-7b-final")

# Measure perplexity on long prompt
long_prompt = "\n".join([schema1, schema2, schema3, question])
inputs = tokenizer(long_prompt, return_tensors="pt").to(model.device)
print(f"Prompt length: {inputs.input_ids.shape[1]} tokens")
with torch.no_grad():
    outputs = model(**inputs, labels=inputs.input_ids)
    print(f"Perplexity: {torch.exp(outputs.loss):.2f}")
```

Note for Qwen2.5 specifically: Qwen2.5 was pretrained with 32K+ tokens and uses "dynamic" rope_scaling — it may already handle 8192 tokens without YaRN fine-tuning. Test the base model first before assuming YaRN is needed.

## Task 4: Decision Matrix

```markdown
## Context Extension Decision

Current context: 4096 tokens
Average schema in Custom-200: ~800 tokens → safe headroom
Largest schema in training set: ~2400 tokens → still fits in 4096

When 4096 becomes a bottleneck:
- Schemas with 15+ tables × 6 columns avg: 15 × 6 × ~15 tokens = 1350 tokens
- Schemas with 30+ tables: 2700+ tokens, leaving 1396 for question + SQL
- Multi-turn: 5 turns × 400 tokens = 2000 tokens history

Recommendation: No YaRN fine-tuning for Week 75 baseline.
- Custom-200 benchmark does not stress 4096-token context.
- Multi-turn (Week 76) is the first scenario where context extension matters.
- Add schema compression preprocessing (select relevant tables) as the primary
  mitigation; add YaRN as a stretch goal in Week 76.
```

## Common Gotchas

- Qwen2.5 has built-in context extension via "dynamic" rope_scaling: check `config.rope_scaling` before applying YaRN — you may find the model already supports longer contexts than you think.
- Flash Attention 2 is required for contexts > 4096 tokens on GPU: without it, attention computation uses O(n²) memory and will OOM at 8192 tokens on a single A100-40GB.
- YaRN fine-tuning data must include both long and short examples: training only on long examples degrades short-context performance.

## How to Verify You Did It Right

Count the tokens in your largest training schema using `tokenizer(schema, return_tensors="pt").input_ids.shape[1]`. If the count is < 3000, your current 4096-token context is sufficient for all your training data and context extension is a nice-to-have, not a necessity. If any schema exceeds 3000 tokens, context extension is worth pursuing.
