# Week 75 Resources — Iteration: Different Base Models

## Papers

[Qwen2.5-Coder Technical Report](https://arxiv.org/abs/2409.12186) — Architecture and training details for the Qwen2.5-Coder family, including code-specialized pretraining data.

[DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) — Original R1 paper; Section 4 covers distillation to 7B/8B models and explains why distilled models retain reasoning structure.

[Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118) — Covers sliding-window attention design, GQA, and the Gemma 2 training recipe.

[Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) — Meta's technical report covering Llama 3.1 8B architecture, RoPE scaling, and instruction-tuning procedure.

[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Reference for understanding how LoRA rank and target module choices interact with different attention architectures.

## Videos

- [Andrej Karpathy — Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Andrej Karpathy — 1h56m. Foundational; useful for understanding how different attention variants build on top of the same core mechanism.
- [How to Compare Language Models Fairly](https://www.youtube.com/watch?v=aircAruvnKk) — Yannic Kilcher — ~30 min. Discusses controlled experimental design for model comparisons; directly applicable to this week's experiment.

## Blog Posts / Articles

[Hugging Face — Chat Templating Guide](https://huggingface.co/docs/transformers/chat_templating) — Official documentation on `apply_chat_template`, with model-specific template examples for Llama, Gemma, and Qwen.

[EleutherAI — Eval Harness Model Comparison Guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md) — Explains how to run the same benchmark across multiple model families consistently.

[Weights & Biases — Comparing Runs Across Experiments](https://docs.wandb.ai/guides/app/features/runs/run-comparison) — How to use W&B's comparison view to overlay loss curves and metrics across your four model runs.

## GitHub Repos

[huggingface/trl](https://github.com/huggingface/trl) — SFTTrainer source; check `DataCollatorForCompletionOnlyLM` for how `response_template` is used per model.

[ggerganov/llama.cpp — Model Support Matrix](https://github.com/ggerganov/llama.cpp/blob/master/README.md) — Verify that your candidate models (Gemma 2, Llama 3.1, DeepSeek-R1-Distill) are supported for downstream GGUF conversion.

[deepseek-ai/DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) — Official repo with distilled model cards and recommended fine-tuning settings.

## Documentation

[Llama 3.1 Model Card — Meta AI](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) — Official HuggingFace model card with chat template specification and recommended generation settings.

[Gemma 2 9B Instruct — Google DeepMind](https://huggingface.co/google/gemma-2-9b-it) — Model card with template format, sliding-window attention notes, and context length.

[DeepSeek-R1-Distill-Qwen-7B Model Card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) — Lists the distillation procedure and recommended system prompt format.

## Optional / Bonus

[Scaling Laws for Neural Language Models (Kaplan et al., 2020)](https://arxiv.org/abs/2001.08361) — Understanding why parameter count alone does not predict task performance; relevant when comparing 7B vs 9B models.

[Are Emergent Abilities of Large Language Models a Mirage? (Schaeffer et al., 2023)](https://arxiv.org/abs/2304.15004) — Critical reading on whether observed performance jumps across architectures are real or metric artifacts; useful calibration for interpreting your comparison results.
