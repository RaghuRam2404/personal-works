# Week 72 Quiz — Frontier Reading: DeepSeek and Qwen

## Multiple Choice

**Q1.** DeepSeek-V3 has 671B total parameters but only 37B active per forward pass. If you fine-tune DeepSeek-Coder-V2-Lite (16B total, 2.4B active) with LoRA rank 64, which parameter count determines the number of trainable LoRA parameters?

A. 16B — LoRA adapters are applied to all stored parameters including inactive experts
B. 2.4B — LoRA adapters are applied only to the active expert parameters in each forward pass
C. The number of LoRA parameters depends on the routing decisions for the training examples
D. Both — LoRA parameters are applied to all experts but only activated experts receive gradient updates

**Q2.** DeepSeek-R1 trains a 671B model with GRPO and then distills the successful reasoning traces into a 7B Qwen model via SFT. What is the fundamental advantage of this distillation approach over training the 7B model with GRPO directly?

A. The 7B model is too small to benefit from GRPO — RL only works at scale
B. Distillation from a much larger, more capable teacher provides higher-quality training signal than the 7B model could generate through its own GRPO exploration
C. GRPO is computationally prohibitive even for 7B models
D. The 7B model lacks the context length needed for GRPO's K-sample generation

**Q3.** Qwen2.5-Coder-7B was already trained with RLHF (DPO on general preference data). Your domain-specific DPO then fine-tuned further on 5K SQL preference pairs. What is the most likely effect of this "DPO on top of RLHF" approach?

A. The second DPO run completely overwrites the first RLHF, making it ineffective
B. The domain DPO shifts the model's preference distribution toward SQL-specific patterns on top of the RLHF-established general helpfulness baseline
C. Two DPO runs cause destructive interference and reduce accuracy
D. The RLHF makes domain DPO unnecessary — Qwen's general preferences cover SQL adequately

**Q4.** Multi-head Latent Attention (MLA) reduces the KV cache by compressing K and V into a low-dimensional latent space. What is the primary inference benefit of this compression?

A. Faster matrix multiplications during the attention computation
B. Smaller KV cache enables longer context at the same VRAM budget, or the same context at lower VRAM cost
C. The latent compression improves model accuracy by removing noise from K and V
D. MLA enables batched inference without any memory overhead

## Short Answer

**Q5.** Explain what "cold start" means in DeepSeek-R1's training and why it is necessary before running GRPO.

**Q6.** You are deciding between Llama 3.1 8B and DeepSeek-R1-Distill-Qwen-7B for Week 75. Both have similar parameter counts. Give two reasons to prefer R1-Distill and two reasons to prefer Llama 3.1.

**Q7.** Qwen2.5's technical report shows Qwen2.5-Coder-7B achieves 79.1% on HumanEval (Python coding benchmark). Explain why strong HumanEval performance is a useful but imperfect predictor of SQL generation quality.

## Deep Scenario

**Q8.** Your colleague proposes applying R1-style reasoning to SQL: instead of training the model to output SQL directly, train it to output a reasoning trace like "The user wants hourly totals → I should use time_bucket with INTERVAL '1 hour' → The schema has a 'created_at' timestamp column → The query is: SELECT time_bucket(...)."

Write a 200-word analysis: (a) identify two benefits of chain-of-thought for SQL generation, (b) identify two challenges specific to SQL that differ from math/code reasoning, and (c) propose how you would create training data for this approach without a 671B teacher model.
