# Week 24 Assignment Solutions

## Task 1 — Architecture Comparison Table (Reference Values)

| Dimension | Llama 3-8B | Qwen2.5-7B | Qwen2.5-Coder-7B | DeepSeek-V3 | DeepSeek-Coder-V2-Lite |
|---|---|---|---|---|---|
| Params (total) | 8B | 7.6B | 7.6B | 671B | 16B |
| Params (active) | 8B (dense) | 7.6B (dense) | 7.6B (dense) | 37B (MoE) | 2.4B (MoE) |
| Training tokens | 15T | 18T | 5.5T | 14.8T | 10.2T |
| Attention type | GQA (8 KV heads) | GQA | GQA | MLA | GQA |
| FFN type | SwiGLU | SwiGLU | SwiGLU | MoE (SwiGLU experts) | MoE |
| Context length | 8K (pretrain), 128K (extended) | 128K | 128K | 128K | 128K |
| Vocabulary size | 128,256 | 151,936 | 151,936 | 129,280 | 102,400 |
| Tokenizer | tiktoken-style | tiktoken-style | tiktoken-style | tiktoken-style | tiktoken-style |
| Special features | RoPE 500K base | Dual chunk attn | FIM training | MLA, FP8 training | FIM, MoE |

Note: values may change with paper updates. Always verify against the specific version of the paper.

---

## Task 4 — Model Selection: Key Points for Grading

**Strongest answer: Qwen2.5-Coder-7B**

**Why Qwen2.5-Coder-7B wins for PostgreSQL text-to-SQL:**

1. **Code-focused pretraining:** 5.5T tokens of code-first data means the model has been exposed to enormous amounts of SQL from GitHub, Stack Overflow, and documentation. It already writes syntactically valid SQL and understands JOIN, subquery, and aggregation patterns.

2. **Right parameter count for fine-tuning:** 7B parameters is the practical sweet spot — large enough to hold complex PostgreSQL knowledge, small enough to fine-tune on a single A100 with QLoRA (Phase 4) and eventually full SFT.

3. **Qwen's strong license:** Apache 2.0 (Qwen2.5 series), allowing commercial use if your project becomes a product.

4. **Active community and tooling:** Strong HuggingFace integration, many fine-tuned variants for reference, good documentation.

**Risks and limitations:**
- Qwen2.5-Coder may have less TimescaleDB-specific training data than general PostgreSQL SQL
- The large vocabulary (151K tokens) means fine-tuning the LM head is expensive if you need new tokens
- Qwen2.5-Coder was trained primarily on Chinese and English — ensure your SQL documentation is English

**Also defensible: DeepSeek-Coder-V2-Lite (16B total, 2.4B active)**
- The MoE architecture means inference cost is similar to a 2.4B model
- Trained on 10.2T tokens with heavy code focus
- Risk: MoE models can be harder to fine-tune (routing instability)

**Common mistakes in student answers:**
- Choosing the largest model (DeepSeek-V3 at 671B total) without considering inference cost — 37B active params requires 4× A100s just for inference
- Choosing Llama 3-8B because it is from Meta — it is excellent general-purpose but has less SQL training than Qwen2.5-Coder
- Not justifying why SQL-specific pretraining matters

---

## How to Verify Your Table is Correct

Cross-check your table values against the model cards on HuggingFace:
- [Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)
- [meta-llama/Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
- [deepseek-ai/DeepSeek-Coder-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Base)

Model cards often summarize the key specs from the paper in a concise table.
