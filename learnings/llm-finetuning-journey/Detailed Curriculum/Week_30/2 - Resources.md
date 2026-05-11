# Week 30 Resources

## Papers

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al. 2021. **Read fully this week.** The foundational LoRA paper; read the math in Section 4 and the ablations in Section 7.
- [Parameter-Efficient Fine-Tuning Methods Survey](https://arxiv.org/abs/2403.14608) — Comprehensive survey of adapter methods including LoRA, prefix tuning, prompt tuning. Skim for context.
- [Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning](https://arxiv.org/abs/2012.13255) — Aghajanyan et al. 2020. The theoretical basis for why LoRA works; read abstract + Section 3.

## Videos

- [LoRA explained](https://www.youtube.com/watch?v=DhRoTONcyZE) — Yannic Kilcher, YouTube, ~25m. Clear walkthrough of the paper math.
- [Practical Tips for Finetuning LLMs Using LoRA](https://www.youtube.com/watch?v=YVU5wAA6Txo) — Sebastian Raschka, YouTube, ~1h. Empirical insights on rank, alpha, target_modules.

## Blog Posts / Articles

- [Sebastian Raschka: Practical Tips for Finetuning LLMs Using LoRA (article version)](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) — The written companion to the video. Dense with empirical findings.
- [LoRA from scratch — Explained](https://lightning.ai/pages/community/tutorial/lora-llm/) — Lightning AI blog. Step-by-step implementation tutorial.

## GitHub Repos

- [LoRA reference implementation (Microsoft)](https://github.com/microsoft/LoRA) — The original authors' repo with example code.
- [peft library (HuggingFace)](https://github.com/huggingface/peft) — The production LoRA implementation you will use in Week 31.

## Documentation

- [PEFT LoRA concept guide](https://huggingface.co/docs/peft/conceptual_guides/lora) — HuggingFace. Explains the math and all hyperparameters.

## Optional / Bonus

- [AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2303.10512) — A variant that automatically determines the optimal rank per layer using importance scoring. Good context for understanding rank ablations.
- [GaLore: Gradient Low-Rank Projection](https://arxiv.org/abs/2403.03507) — A newer approach (2024) that applies low-rank projection to gradients rather than weights; interesting counterpoint to LoRA.
