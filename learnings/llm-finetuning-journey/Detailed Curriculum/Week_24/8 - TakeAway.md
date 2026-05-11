# Week 24 TakeAway — SOTA Pretraining Recipes

**One-liner:** Qwen2.5-Coder-7B (5.5T code tokens) is your fine-tuning base; MoE enables large total params with small active compute; GQA shrinks KV cache.

---

## Model Cheat Sheet

| Model | Params (active) | Train Tokens | Best For |
|---|---|---|---|
| Llama 3-8B | 8B | 15T | General purpose, strong instruction |
| Qwen2.5-7B | 7.6B | 18T | Math, reasoning, multilingual |
| Qwen2.5-Coder-7B | 7.6B | 5.5T (code) | Code gen, SQL, fine-tuning base |
| DeepSeek-V3 | 37B active / 671B total | 14.8T | GPT-4-class performance, low inference cost |
| DeepSeek-Coder-V2-Lite | 2.4B active / 16B total | 10.2T | Code, MoE, low inference cost |

---

## Architecture Innovations

| Innovation | What it does | Where |
|---|---|---|
| GQA | Fewer KV heads → smaller KV cache | Llama 3, Qwen2.5 |
| MoE | Only N experts active per token | DeepSeek-V3, DS-Coder-V2 |
| MLA | Compressed KV cache via latent projections | DeepSeek-V3 |
| FIM training | Bidirectional code infilling | DeepSeek-Coder, Qwen2.5-Coder |
| SwiGLU | Better activation than GeLU for LLMs | All 5 models |
| RoPE 500K base | Long context extrapolation | Llama 3 |

---

## Decision Rules

- For SQL fine-tuning at 7B scale → Qwen2.5-Coder-7B
- For lowest inference cost at GPT-4 quality → DeepSeek-V3 (37B active)
- For strongest general reasoning → Llama 3-70B or Qwen2.5-72B
- For MoE fine-tuning (experimental) → DeepSeek-Coder-V2-Lite (2.4B active)
- License: Qwen2.5 is Apache 2.0; Llama 3 has Meta's Llama license (non-commercial variants restricted)

---

## Post-Training Pipeline (Common Pattern)

```
Base pretrained model
  → Stage 1: SFT on instruction dataset (format learning)
  → Stage 2: DPO or GRPO (alignment / quality improvement)
  → Stage 3: Specialized SFT (code, math, tools)
  → Final: Safety fine-tuning / RLHF
```

---

## Numbers to Remember

| Model | Vocab Size | Context Length | Architecture |
|---|---|---|---|
| Llama 3-8B | 128,256 | 128K | Dense GQA |
| Qwen2.5-Coder-7B | 151,936 | 128K | Dense GQA |
| DeepSeek-V3 | 129,280 | 128K | MoE + MLA |
