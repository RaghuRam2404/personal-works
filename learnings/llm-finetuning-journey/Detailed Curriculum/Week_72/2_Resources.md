# Week 72 Resources — Frontier Reading: DeepSeek and Qwen

## Papers (primary reading this week)

[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — DeepSeek-AI 2024; MoE architecture, MLA, FP8 training, and frontier benchmark results; focus on Sections 2 (architecture) and 3 (training).

[DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) — DeepSeek-AI 2025; three-stage pipeline, GRPO details, and distillation into 7B Qwen model; focus on Sections 2 (method) and 4 (results).

[Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115) — Qwen Team 2024; data mixture, architecture, Coder and Math variants, and post-training details; focus on Sections 3 (data) and 5 (evaluation).

## Supporting Papers

[DeepSeek-Coder-V2: Breaking the Barrier of Closed-Source Models in Code Intelligence](https://arxiv.org/abs/2406.11931) — DeepSeek-AI 2024; the technical report for DeepSeek-Coder-V2-Lite specifically; SQL benchmark results are here.

[Qwen2.5-Coder Technical Report](https://arxiv.org/abs/2409.12186) — Hui et al. 2024; the Coder-specific technical report with SQL/code benchmark details separate from the main Qwen2.5 report.

## Videos

[DeepSeek-R1 Explained (Yannic Kilcher)](https://www.youtube.com/watch?v=lMLwS52TbMs) — ~40 min; detailed walkthrough of the three-stage pipeline and GRPO reward design.

[DeepSeek-V3 Architecture Breakdown (Trelis Research)](https://www.youtube.com/watch?v=Q5pqVOqxpVk) — ~30 min; MoE routing and MLA explained with diagrams.

## Blog Posts / Articles

[DeepSeek-R1 Zero: Reasoning Through RL Without SFT (DeepSeek blog)](https://api-docs.deepseek.com/news/news250120) — The DeepSeek team's own overview of R1 training.

[Understanding DeepSeek's MoE Architecture (Sebastian Raschka)](https://magazine.sebastianraschka.com/p/understanding-deepseeks-mixture-of) — Clear explanation of expert routing and load balancing.

## GitHub Repos

[deepseek-ai/DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) — Model cards and evaluation scripts; the distilled 7B models are accessible here.

[QwenLM/Qwen2.5-Coder](https://github.com/QwenLM/Qwen2.5-Coder) — Training details, demos, and the model family tree showing how Coder relates to the base Qwen2.5.

[unslothai/unsloth — supported models](https://github.com/unslothai/unsloth?tab=readme-ov-file#-finetune-for-free) — Check this before committing to any Week 75 base model; Unsloth support determines your training toolchain.

## Optional / Bonus

[DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434) — The predecessor to V3; explains MLA in more detail since V3 assumes familiarity.

[Scaling Laws for Mixture of Experts Language Models (Artetxe et al.)](https://arxiv.org/abs/2112.02099) — Theoretical background on how MoE models scale differently from dense models; useful context for the V3 paper.
