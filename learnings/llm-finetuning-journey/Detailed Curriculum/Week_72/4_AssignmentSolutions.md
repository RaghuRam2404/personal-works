# Week 72 Assignment Solutions

## Task 1: MLA and MoE Explained

MLA (Multi-head Latent Attention) compresses the KV cache by projecting keys and values into a low-dimensional latent space before caching. Instead of caching full K and V matrices (d_model × n_heads × seq_len), it caches a compressed representation of dimension d_latent (typically d_model / 8 to d_model / 16). At inference, K and V are reconstructed from the latent representation on the fly. The memory saving at 4096 context: standard GQA with 8 groups for a 7B model uses ~0.8 GB for KV cache; MLA reduces this to ~0.1 GB — an 8x reduction. The accuracy tradeoff is small (the latent projection is learned, not a lossy compression) — DeepSeek reports < 0.1 pp degradation vs full attention.

MoE routing: DeepSeek-V3 has 256 expert FFN sub-networks per layer. For each token, the router selects 8 active experts (Top-K routing with K=8). The routing function is a learned linear projection from the token embedding to expert logits, followed by softmax and Top-K selection.

## Task 2: R1 vs Your GRPO

R1's three stages in 2 sentences each:

Cold start: the team generates a small set (a few hundred) of long chain-of-thought examples with explicit reasoning traces, then SFT on these to give the model a "template" for reasoning before RL. Without cold start, GRPO from scratch produces incoherent intermediate steps.

GRPO RL: the model generates K=8–64 completions per problem, scores them with verifiable rewards (math equality, code execution), and trains via the Group Relative Policy Optimization loss. This continues for thousands of steps across math, code, and logic tasks.

Rejection sampling + distillation: the best GRPO-generated traces (those achieving the highest reward) are collected and used as SFT data to train smaller models (R1-Distill-Qwen-7B). The small model never runs GRPO itself — it learns from the large model's successful reasoning traces.

Comparison to your GRPO: you ran K=8, 600 steps, SQL execution reward. R1 ran K=8–64, thousands of steps, multi-task rewards. Your GRPO is a domain-specific version of R1's Stage 2.

## Task 3: Qwen2.5 Knowledge Gaps

Qwen2.5-Coder's SFT data likely includes standard SQL (Spider-style) but not TimescaleDB-specific patterns. This explains why zero-shot performance on your Custom-200 was only 61% — the model knows SQL syntax but not the TimescaleDB API.

On the RLHF-vs-domain-DPO interaction: Qwen2.5-Instruct was trained with general RLHF that optimized for helpfulness and harmlessness. This means the base model is slightly biased toward explanatory, conversational responses. Your domain DPO (which penalizes non-SQL output) is fighting against this bias — which is why DPO took 800 steps to converge rather than fewer.

## Task 4: Decision Matrix Template

```markdown
| Model | Active Params | Spider EM | SQL Tokens | Unsloth | Quant | Verdict |
|-------|-------------|-----------|-----------|---------|-------|---------|
| Llama 3.1 8B | 8B | ~72% | Good | Yes | Simple | Candidate |
| Gemma 2 9B | 9B | ~74% | Good | Yes | Simple | Candidate |
| DeepSeek-Coder-V2-Lite | 2.4B | ~80%+ | Excellent | Partial | Complex | Risky |
| R1-Distill-Qwen-7B | 7B | ~78%? | Good | Yes | Simple | Recommended |
```

Recommendation: R1-Distill-Qwen-7B. Same Qwen2.5 architecture as your existing pipeline (full Unsloth support, simple quantization), but with stronger chain-of-thought reasoning from R1 distillation that may help on complex multi-table SQL. DeepSeek-Coder-V2-Lite is compelling on benchmarks but MoE quantization complexity is a practical barrier for Week 75's timeline.

## Common Gotchas

- MoE model quantization: GPTQ and AWQ work on MoE models but require more VRAM during the calibration pass because all expert weights must be loaded simultaneously. Budget 50% more VRAM than you would expect from active parameter count.
- R1-Distill-Qwen-7B was fine-tuned from Qwen2.5-Math-7B, not Qwen2.5-Coder-7B. The base is math-optimized, which means it may have slightly weaker initial SQL syntax knowledge — but stronger multi-step reasoning.
- DeepSeek-V3's MLA is not implemented in standard HuggingFace transformers; it requires the DeepSeek custom modeling code. Verify compatibility before Week 75.

## How to Verify You Did It Right

Your decision matrix is complete if: (1) every cell has a source (paper section or measured value), (2) your recommendation is justified by the matrix — not by gut feeling, and (3) you have identified the one biggest risk of your recommended model.
