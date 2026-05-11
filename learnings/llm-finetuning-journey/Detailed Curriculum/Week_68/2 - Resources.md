# Week 68 Resources — Training Pipeline Documentation

## Papers

[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al. 2022; cite this for all LoRA architecture choices; the alpha/rank relationship is explained here.

[QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Dettmers et al. 2023; cite for alpha=2×rank convention and 4-bit NF4 quantization during training.

[Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) — Rafailov et al. 2023; cite for DPO loss formula and β interpretation.

[DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) — Shao et al. 2024; the GRPO paper; cite for your Group Relative Policy Optimization training stage.

[Qwen2.5-Coder Technical Report](https://arxiv.org/abs/2409.12186) — Hui et al. 2024; cite as the base model; include the exact arXiv ID.

## Videos

[How to Write a Machine Learning Paper — Methods Section (Yannic Kilcher)](https://www.youtube.com/watch?v=UPkNSWmY89c) — ~25 min; practical advice on describing training procedures clearly.

## Blog Posts / Articles

[Weights & Biases — Logging Hyperparameters](https://docs.wandb.ai/guides/track/config) — How to ensure your training config is fully logged for later extraction into the paper.

[ML Reproducibility Checklist (NeurIPS 2021)](https://neurips.cc/Conferences/2021/PaperInformation/PaperChecklist) — The formal checklist peer reviewers use; run your training section through it.

## GitHub Repos

- [huggingface/transformers](https://github.com/huggingface/transformers) — HuggingFace. The core library used throughout your training pipeline; link directly to the version tag (e.g., `v4.40.0`) pinned in your `requirements.txt` so reviewers can reproduce your exact environment.
- [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — EleutherAI. The evaluation framework used in Week 23 and throughout Phase 4; include the commit SHA you used in your technical report for reproducibility.
- [allenai/OLMo](https://github.com/allenai/OLMo) — Allen AI. One of the most fully-documented open training pipelines available; use as a reference for how to document your own training pipeline choices, hyperparameter tables, and compute budget in your technical report.

## Documentation

[HuggingFace Trainer — All Hyperparameters](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments) — Full list of TrainingArguments fields; use to verify you have documented all non-default values.

[TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer) — SFT-specific arguments; covers `max_seq_length`, `packing`, `dataset_text_field`.

[TRL DPOTrainer](https://huggingface.co/docs/trl/dpo_trainer) — DPO-specific arguments; covers `beta`, `loss_type`, `reference_model`.

## Optional / Bonus

[Chinchilla Scaling Laws for Neural Language Models](https://arxiv.org/abs/2203.15556) — Hoffmann et al. 2022; cite in your compute section if you want to contextualize why fine-tuning 12 GPU-hours suffices vs pre-training thousands.

[The Illustrated Transformer Training (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/) — Useful for explaining to a general ML audience what continued pretraining does to representations.
