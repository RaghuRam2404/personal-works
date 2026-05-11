# Week 72 TakeAway — Frontier Reading: DeepSeek and Qwen

R1 validates your GRPO; V3's MLA is the future of KV cache; Qwen2.5-Coder is a strong base and you already know why your DPO helped.

## Key Architectural Facts

```
DeepSeek-V3:  671B total, 37B active, 256 experts, Top-8 routing, MLA KV compression
DeepSeek-R1:  Cold start → GRPO (K=8-64) → Rejection sampling → Distill to 7B
Qwen2.5-Coder-7B: 18T token pretraining, code/SQL-optimized SFT, RLHF DPO post-training
```

## Base Model Decision Matrix (for Week 75)

| Model | Active Params | Toolchain | SQL Baseline | Quant | Recommendation |
|-------|-------------|-----------|-------------|-------|----------------|
| Llama 3.1 8B | 8B | Full | Good | Simple | Safe choice |
| Gemma 2 9B | 9B | Full | Good | Simple | Safe choice |
| DeepSeek-Coder-V2-Lite | 2.4B | Partial | Excellent | Complex | High risk |
| R1-Distill-Qwen-7B | 7B | Full | Good+ | Simple | Best tradeoff |

## Decision Rules

- If base model has MoE architecture: verify Unsloth supports it before committing to Week 75
- If base model was fine-tuned from Math variant: add extra SQL CPT to compensate
- If choosing between similar-quality models: pick the one with better toolchain support
- R1-style reasoning for SQL: use GPT-4o to generate 500 CoT SQL traces, verify executability, add to SFT
- DPO on top of RLHF: works; domain DPO shifts only SQL-prompt distribution, not general behavior

## Numbers to Remember

- DeepSeek-V3: 37B active / 671B total; MLA = ~8x KV cache compression
- R1 cold start: a few hundred CoT demonstrations are enough to seed reasoning behavior
- Qwen2.5-Coder pretraining: 18T tokens with code emphasis; zero-shot SQL ≈ 61–76% on Spider
- Distillation ratio: 671B teacher → 7B student via rejection sampling on verified reasoning traces

## Red Flags

- Assuming V3's 671B results transfer to Lite 16B: scale matters — verify on your benchmark
- Using DeepSeek-Coder-V2-Lite without verifying Unsloth support: likely to hit implementation blockers
- Expecting R1-Distill-Qwen-7B to have strong SQL syntax from day 1: it was Math-distilled, not Coder — add CPT
- Running GRPO without cold start on a new base model: bootstrap failure risk — always seed with SFT first
