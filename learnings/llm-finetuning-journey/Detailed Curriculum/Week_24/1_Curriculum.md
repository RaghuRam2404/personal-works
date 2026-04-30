# Week 24 — Reading Week: SOTA Pretraining Recipes (2024–2025)

## Learning Objectives

By the end of this week, you will be able to:

- Identify the key architectural and training differences between Llama 3, Qwen2.5, Qwen2.5-Coder, DeepSeek-V3, and DeepSeek-Coder
- Explain what "post-training" pipeline means and how it differs across these models
- Compare their data strategies (sources, mixing ratios, filtering approaches)
- Summarize the compute scales and efficiency innovations used in each recipe
- Form a reasoned opinion on which base model is the best starting point for your PostgreSQL fine-tuning goal

---

## Concepts

### Why Reading Weeks Matter

Reading papers is a skill. Most engineers read only abstracts and conclusions; senior engineers read selectively but deeply. This week, you practice reading 5 major technical reports at different depths.

**Your reading strategy:**
1. Skim all 5 (30 min each): abstract + intro + architecture table + training setup + key results table
2. Deep-read 2 (1–2 hours each): the models most relevant to your goal (Qwen2.5-Coder, DeepSeek-Coder)
3. Synthesize: write a comparison document that would help a colleague choose between these models

### Llama 3 — Meta's 2024 Foundation Model

Paper: [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)

**Key facts:**
- Sizes: 8B, 70B, 405B
- Training data: 15T tokens (heavily curated, multilingual, code-heavy)
- Architecture: Standard GQA (Grouped Query Attention) transformer; 128K context via RoPE
- Tokenizer: 128K vocabulary (tiktoken-style BPE)
- Post-training: SFT + RLHF + DPO + specialized code/math/tool-use fine-tuning
- Key insight: Meta's biggest advance was in data curation — aggressive filtering, quality scoring with LLM judges, and synthetic data generation

**Why Llama 3 matters for you:**
Llama 3 8B is the baseline strong general-purpose model at the 7–8B scale. Its 15T-token training means it is highly over-trained by Chinchilla standards, making it an excellent base for fine-tuning because the model has dense, well-calibrated representations.

**Architecture highlights:**
- GQA with 8 KV heads instead of 32 — reduces KV cache size by 4× during inference
- RoPE rotary embeddings with 500,000 base frequency — enables long context extrapolation
- No bias terms in attention and MLP layers
- SwiGLU activation (combines Swish and gated linear unit) instead of GeLU

### Qwen2.5 — Alibaba's Dense Model Family

Paper: [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)

**Key facts:**
- Sizes: 0.5B to 72B
- Training data: 18T tokens; heavy math and coding emphasis
- Key innovations: long-context pretraining (up to 128K), dual chunk attention for long context
- Tokenizer: 151,936 vocabulary (tiktoken, same as GPT-4)
- Post-training: SFT → DPO → GRPO (Group Relative Policy Optimization)

**What Qwen2.5 does well:**
- Math and reasoning benchmarks (top-tier at the 7B scale)
- Long-context tasks due to dedicated long-context training
- Multilingual (including Chinese, Japanese, Korean)

**Why Qwen2.5 matters for your project:**
The base Qwen2.5-7B is a strong general model. But for your PostgreSQL use case, Qwen2.5-Coder is the more relevant variant.

### Qwen2.5-Coder — Alibaba's Code Model

Paper: [Qwen2.5-Coder Technical Report](https://arxiv.org/abs/2409.12186)

**Key facts:**
- Sizes: 1.5B to 32B
- Training data: 5.5T tokens, heavily code-first (including SQL, Bash, Python, Java, etc.)
- Built on top of Qwen2.5 base; continued pretraining on code-focused data
- Code-specific features: repository-level context during training, fill-in-the-middle (FIM) training

**Why Qwen2.5-Coder is your best starting point:**
Qwen2.5-Coder-7B has already seen enormous amounts of SQL from GitHub and Stack Overflow. It can write syntactically valid SQL out of the box. Your fine-tuning will amplify its existing SQL knowledge and specialize it for PostgreSQL/TimescaleDB idioms — rather than teaching SQL from scratch.

**Qwen2.5-Coder vs Llama 3 for SQL:**
Qwen2.5-Coder-7B trained on 5.5T code-focused tokens will outperform Llama 3-8B (15T general tokens) on SQL generation tasks at the same parameter count. The code-focused pretraining gives it a head start on SQL syntax, schema understanding, and code reasoning.

### DeepSeek-V3 — DeepSeek's MoE Model

Paper: [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)

**Key facts:**
- Architecture: Mixture of Experts (MoE) — 671B total params, 37B active per token
- Training data: 14.8T tokens
- Key innovations: Multi-Head Latent Attention (MLA) — compresses KV cache; FP8 training — 2× memory efficiency
- Achieves GPT-4-class performance at much lower inference cost (only 37B params active)

**MoE basics (you need to understand this):**
In a MoE transformer, the FFN layer is replaced with N "experts" (smaller FFNs), and a router selects 2–4 experts per token. Only selected experts are activated — hence 37B active out of 671B total. Benefits:
- More parameters for the same compute
- Different experts specialize in different domains

**What to read in DeepSeek-V3:**
Focus on Section 2 (architecture), Section 3 (training data), and the engineering section (FP8 training, gradient communication). The MLA attention optimization is particularly interesting if you want to understand KV cache compression.

### DeepSeek-Coder — Code-Specialized MoE Model

Paper: [DeepSeek-Coder Technical Report](https://arxiv.org/abs/2401.14196)

**Key facts:**
- Released early 2024; precursor to DeepSeek-V3
- Trained on 2T tokens with 87% code and 13% natural language
- Key insight: the fill-in-the-middle (FIM) training objective improves code completion quality
- The PSM (Prefix-Suffix-Middle) format: model predicts the middle of a code snippet given prefix and suffix

**Fill-in-the-middle training:**
```
Input format: <prefix>code_prefix<suffix>code_suffix<middle>
Target:       <middle>code_middle</s>
```

During FIM, the model learns to infill code from both directions, making it better at code completion tools (where you have surrounding context).

**For your project:** DeepSeek-Coder-V2-Lite is a compelling alternative to Qwen2.5-Coder-7B. Both are strong code models; Qwen2.5-Coder has the advantage of more recent training data.

### What to Look for in Each Paper

For your 3-page comparison document, focus on these dimensions for each model:

| Dimension | What to extract |
|---|---|
| Data | Token count, sources, mixing ratios, filtering approach |
| Architecture | Attention type, FFN type, context length, vocab size, special features |
| Training recipe | LR schedule, optimizer, batch size, warmup |
| Post-training | Which methods: SFT, DPO, RLHF, GRPO, RLVR |
| Evaluation | Which benchmarks, how they compare to each other |
| Efficiency tricks | FP8, MoE, GQA, KV cache compression |

---

## Connections

**Prior weeks (17–23):** Your pretraining experience makes the paper sections on training data, optimization, and evaluation directly accessible. You now understand what "18T tokens" means in practice.

**Phase 4+:** Qwen2.5-Coder-7B or DeepSeek-Coder-V2-Lite will be your fine-tuning base. Your Week 24 reading informs that choice.

---

## Common Misconceptions

- **"The model with the most parameters wins."** DeepSeek-V3 at 671B params (37B active) vs. Llama 3-70B: DeepSeek-V3 wins on many benchmarks at lower active-parameter cost. More params are not always better.
- **"I should fine-tune the largest model I can access."** Larger models are harder to fine-tune, more expensive to serve, and often over-parameterized for narrow tasks. 7B is the right size for your use case.
- **"Post-training (SFT, DPO) is just fine-tuning."** Post-training is a complex multi-stage pipeline — SFT teaches format, DPO/GRPO aligns with human preferences, and specialized stages add capabilities (code, math, tools). Understanding this pipeline is what Phase 4–6 is about.

---

## Time Allocation (6–8 hrs)

- 1h: Skim Llama 3 paper (focus on data, architecture table, benchmark table)
- 1.5h: Deep-read Qwen2.5-Coder paper (all sections)
- 1.5h: Deep-read DeepSeek-Coder paper (all sections)
- 1h: Skim Qwen2.5 and DeepSeek-V3 papers
- 2h: Write the 3-page comparison document (`week-24-sota-comparison.md`)
