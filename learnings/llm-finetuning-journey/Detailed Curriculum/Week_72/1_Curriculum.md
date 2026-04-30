# Week 72 — Frontier Reading 2: DeepSeek and Qwen Technical Reports

## Learning Objectives

By the end of this week, you will be able to:

- Explain the Mixture-of-Experts (MoE) architecture used in DeepSeek-V3 and why it enables frontier capability at lower active parameter counts
- Describe DeepSeek-R1's chain-of-thought reinforcement learning approach and relate it to your GRPO training
- Summarize Qwen2.5's training pipeline and identify the specific decisions relevant to SQL code generation
- Evaluate whether switching to a DeepSeek or Qwen base model for Week 75 is motivated by these papers
- Extract three concrete techniques to apply in your iteration weeks

## Concepts

### DeepSeek-V3: Mixture of Experts at Frontier Scale

DeepSeek-V3 (DeepSeek-AI, 2024) is a 671B total parameter MoE model with only 37B active parameters per forward pass. Its significance for your work is not the scale — you cannot train or run it — but the architectural and training techniques it introduces that are now trickling into smaller open models.

The key MoE mechanism: instead of all feed-forward network (FFN) layers processing every token through a single FFN, MoE models have multiple "expert" FFN sub-networks and a learned routing mechanism that selects K experts (typically K=2 or 8) for each token. Only the selected experts' weights are loaded and computed — inactive experts consume no FLOPs. This allows a model with 671B stored parameters to run at 37B FLOPs, achieving near-frontier capability at much lower inference cost.

For your SQL model: DeepSeek's MoE approach is directly relevant to Week 75, where DeepSeek-Coder-V2-Lite (16B total, 2.4B active) is a candidate base model. Lite models use the same MoE architecture as the full model, meaning you get routing efficiency at a size that fits on a single A100-40GB.

DeepSeek-V3's training innovations that transfer to smaller models:
- Multi-head Latent Attention (MLA): compresses KV cache by projecting keys and values into a low-dimensional latent space, reducing memory ~5x vs standard attention. Directly enables longer context at lower cost.
- FP8 training: using 8-bit floating point for forward passes to reduce memory and increase throughput. Unsloth implements this for supported hardware.
- Auxiliary-loss-free load balancing: earlier MoE models needed a separate loss term to prevent routing collapse (all tokens going to the same expert). DeepSeek-V3 introduces a bias-based balancing mechanism that is simpler and more stable.

### DeepSeek-R1: Chain-of-Thought Reasoning via Reinforcement Learning

DeepSeek-R1 (DeepSeek-AI, 2025) is a reasoning model trained entirely through reinforcement learning — no supervised chain-of-thought demonstrations. Its training process has three stages:

1. Cold start: a small number of long-form chain-of-thought examples to initialize coherent reasoning.
2. RL with verifiable rewards: GRPO on math and code tasks with exact-answer verification. The model learns to generate reasoning traces through trial and error.
3. Rejection sampling + SFT distillation: the best RL-generated reasoning traces are distilled into smaller models (DeepSeek-R1-Distill-Qwen-7B) through SFT.

This is directly relevant to your work in two ways: First, R1's GRPO training validates the approach you took in Week 60 — verifiable rewards enable sophisticated reasoning even without supervised examples. Second, distillation of R1's reasoning into smaller models (Qwen-7B) creates a base model that has strong chain-of-thought capabilities, which benefits SQL generation tasks that require multi-step reasoning (joins across multiple tables, subqueries, window functions).

DeepSeek-R1-Distill-Qwen-7B is worth considering as a base model for Week 75 because it was fine-tuned from Qwen2.5-Math-7B with R1's reasoning traces — it has stronger step-by-step reasoning than the base Qwen2.5-Coder model.

### Qwen2.5: Domain-Specialized Models

Qwen2.5 (Qwen Team, 2024) is the model family you used as your base. Reading the technical report now — after building your full pipeline — lets you understand what the base model already knows and where your fine-tuning adds value.

Key Qwen2.5 training details:

Qwen2.5 was pretrained on 18T tokens, significantly more than its predecessors. The data mixture includes 5–6% code content (estimated), which explains why Qwen2.5-Coder achieves strong SQL baselines before any domain fine-tuning.

The Qwen2.5 series includes domain-specialized variants: Qwen2.5-Coder-7B (code-focused), Qwen2.5-Math-7B (math-focused), and the Instruct variants of each. Your choice of Qwen2.5-Coder-7B-Instruct as the base was appropriate — the Coder variant includes code execution training and SQL-pattern exposure during its specialized fine-tuning.

Qwen2.5's post-training pipeline uses SFT followed by DPO with RLHF preference data, similar to your pipeline. The difference: Qwen uses a large general-purpose preference dataset; you use a small domain-specific verified dataset. Your 5K verified SQL preference pairs likely have higher domain signal than Qwen's general dataset, which is why domain-specific DPO added value on top of the already-DPO'd Instruct model.

### MoE vs Dense Models for SQL Fine-Tuning

A practical question for Week 75: should you switch to a MoE base model (DeepSeek-Coder-V2-Lite, 16B/2.4B active) or stay with a dense model (Qwen2.5-Coder-7B, 7B active)?

Arguments for MoE:
- 2.4B active parameters vs 7B: faster inference despite 16B total parameters
- Expert routing may naturally specialize SQL experts during fine-tuning
- DeepSeek-Coder-V2-Lite has stronger SQL benchmarks than Qwen2.5-Coder-7B at similar inference cost

Arguments for dense:
- Quantization is simpler for dense models: MoE models have irregular memory access patterns that make GGUF quantization less efficient
- LoRA fine-tuning on MoE is less studied; applying LoRA to only active expert FFN layers requires careful implementation
- Your existing pipeline (Unsloth) has better support for Qwen/Llama architecture than DeepSeek MoE

## Connections

These papers contextualize the base model decision you will make in Week 75. DeepSeek-R1-Distill-Qwen-7B is a compelling Week 75 candidate because it combines Qwen2.5's architecture (fully supported by your toolchain) with chain-of-thought reasoning capabilities that may help with complex multi-table SQL. The Qwen2.5 technical report explains what your current base model already knows, helping you avoid redundant CPT and target your fine-tuning more precisely.

## Common Misconceptions / Pitfalls

DeepSeek-V3's scale results (671B parameters) do not directly predict how DeepSeek-Coder-V2-Lite (16B) will perform on your SQL task. The architectural techniques transfer but the capability level scales differently.

R1's GRPO training was done with K=8–64 samples per prompt over many thousands of steps. Your K=8 over 600 steps is a much smaller GRPO run — the improvement patterns may differ.

## Time Allocation (6–8 hours)

- 1.5h: Read DeepSeek-V3 technical report (focus: MLA, MoE routing, training efficiency)
- 1.5h: Read DeepSeek-R1 (focus: cold start, GRPO, distillation to 7B)
- 1.0h: Read Qwen2.5 technical report (focus: data mixture, Coder variant, post-training)
- 1.5h: Write synthesis notes and base model decision matrix for Week 75
- 0.5h: Update `reading_notes/week72_synthesis.md`
