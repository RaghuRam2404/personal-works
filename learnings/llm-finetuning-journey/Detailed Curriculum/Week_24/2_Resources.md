# Week 24 Resources — SOTA Pretraining Recipes

## Papers (Required Reading)

- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) — Meta AI 2024. The full Llama 3 technical report; focus on Sections 2–4 (data, architecture, training).
- [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115) — Alibaba Qwen Team 2024. The Qwen2.5 base model family; Sections 2–3.
- [Qwen2.5-Coder Technical Report](https://arxiv.org/abs/2409.12186) — Alibaba Qwen Team 2024. Deep-read this one for your fine-tuning base understanding; all sections.
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — DeepSeek AI 2024. Focus on architecture innovations (MLA, MoE), FP8 training, and benchmark results.
- [DeepSeek-Coder: When the Large Language Model Meets Programming](https://arxiv.org/abs/2401.14196) — Guo et al. 2024. The original DeepSeek-Coder paper; Sections 2–3 on data and FIM.

## Videos

- [Andrej Karpathy — State of GPT](https://www.youtube.com/watch?v=bZQun8Y4L2A) — Andrej Karpathy — ~45 min. Microsoft Build 2023 talk covering the full post-training pipeline (SFT, reward modeling, RLHF); the most concise explanation of what happens after pretraining.
- [DeepSeek-R1 paper walkthrough](https://www.youtube.com/watch?v=EGFb1pRxiIU) — Yannic Kilcher (~45 min). Background on DeepSeek's approach to reasoning and alignment.
- [Qwen2.5 announcement](https://www.youtube.com/watch?v=YVf8wgmLQCE) — Alibaba Cloud. Short overview (~15 min).

## Blog Posts / Articles

- [Llama 3 model card](https://huggingface.co/meta-llama/Meta-Llama-3-8B) — HuggingFace. Quick reference for architecture specs.
- [Qwen2.5-Coder model card](https://huggingface.co/Qwen/Qwen2.5-Coder-7B) — HuggingFace. Architecture, training data summary, and benchmark table.
- [DeepSeek-V3 model card](https://huggingface.co/deepseek-ai/DeepSeek-V3) — HuggingFace. The most concise summary of the 671B model.
- [Nathan Lambert's post on open LLM training recipes](https://www.interconnects.ai/p/llm-training-data-2024) — Interconnects. Survey of what is and is not disclosed in modern LLM technical reports.

## GitHub Repos

- [QwenLM/Qwen2.5-Coder](https://github.com/QwenLM/Qwen2.5-Coder) — Official Qwen2.5-Coder repo; includes inference examples and fine-tuning guides.
- [deepseek-ai/DeepSeek-Coder](https://github.com/deepseek-ai/DeepSeek-Coder) — Official DeepSeek-Coder repo; includes FIM examples and model conversion scripts.
- [meta-llama/llama-recipes](https://github.com/meta-llama/llama-recipes) — Meta's official fine-tuning and inference recipes for Llama 3.

## Documentation

- [Qwen2.5 documentation](https://qwen.readthedocs.io/en/latest/) — Official docs for Qwen models, including fine-tuning guides.
- [HuggingFace model hub — code models leaderboard](https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard) — Rankings of code models; compare Qwen2.5-Coder and DeepSeek-Coder on HumanEval, MBPP.

## Optional / Bonus

- [Mixtral of Experts](https://arxiv.org/abs/2401.04088) — Jiang et al. 2024, Mistral. The MoE architecture paper from a Western lab; good complement to DeepSeek-V3's MoE description.
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) — Ainslie et al. 2023. The original GQA paper; explains the attention efficiency improvement.
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — Su et al. 2021. The RoPE paper; understanding this helps you read Llama 3's architecture section.
