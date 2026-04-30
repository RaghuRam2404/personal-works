# Week 74 TakeAway — Context Extension: LongRoPE and YaRN

RoPE breaks at long contexts because rotation angles go out-of-distribution; YaRN fixes this cheaply (400 steps); LongRoPE fixes it optimally (100+ GPU-hours).

## Key Concepts

```
RoPE:         Q_m · K_n = f(m-n, {θ_i}) — relative position via rotation
Out-of-dist:  positions > train_len produce unseen rotation angles → degraded attention
Interpolation: m → m × (old/new) — fixes OOD but compresses high-freq dims
YaRN:         non-uniform interpolation: skip high-freq, scale only low-freq + temperature
LongRoPE:     evolutionary search for optimal per-dim scaling; pays 50–200 GPU-h
```

## YaRN Config for Qwen2.5

```python
config.rope_scaling = {
    "type": "yarn",
    "factor": 2.0,          # 4096 → 8192 tokens
    "original_max_position_embeddings": 4096,
}
# Requires: attn_implementation="flash_attention_2" for long sequences
# Fine-tune: 400 steps on mixture of long + short examples
```

## Decision Rules

- If schema < 3000 tokens: no context extension needed — 4096 is sufficient
- If schema 3000–6000 tokens: apply schema compression (select top-5 relevant tables)
- If schema > 6000 tokens or multi-turn > 3000 tokens: add YaRN fine-tuning (0.5 GPU-hours)
- If extending to > 32K tokens: use LongRoPE; YaRN quality degrades beyond 4x extension
- Always use Flash Attention 2 for contexts > 4096 tokens: O(N) vs O(N²) memory

## Numbers to Remember

- YaRN fine-tuning: 400 steps; ~0.5 GPU-hours for 2x extension
- LongRoPE search: 50–200 GPU-hours; worthwhile only for 32K+ extension
- Flash Attention 2 memory: O(N) vs standard O(N²); 8192 tokens fits in 40 GB with FA2
- Qwen2.5 native context: 32K tokens (check config before assuming 4096 is the limit)
- Schema compression: top-5 tables covers >90% of typical SQL queries

## Red Flags

- Extending context without Flash Attention 2: OOM at any useful long context
- YaRN without fine-tuning: perplexity OK, SQL accuracy may still degrade 10+ pp
- LongRoPE's evolutionary search on a 2h compute budget: use YaRN instead
- Schema compression without a test for cross-table queries: compression will silently fail on JOINs across filtered tables
- Assuming Qwen2.5 needs context extension: it has 32K native context — test first
