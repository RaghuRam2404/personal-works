# Week 35 Resources

## Papers

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al. 2021. Re-read Section 7 for the rank ablations — directly relevant to this week's sweep.
- [NEFTune: Noisy Embeddings Improve Instruction Finetuning](https://arxiv.org/abs/2310.05914) — Jain et al. 2023. The noise embedding technique available via `neftune_noise_alpha` in SFTConfig.

## Videos

- [LoRA Insights: Rank, Alpha, and Target Modules from 100+ Experiments](https://www.youtube.com/watch?v=XpoKB3usmKc) — Sebastian Raschka — ~35 min. Empirical analysis of how learning rate, batch size, rank, and packing interact in LoRA fine-tuning; directly applicable to this week's hyperparameter sweep.
- [AdamW and Weight Decay: Why It Matters for Fine-Tuning](https://www.youtube.com/watch?v=0xC_G1ky8-4) — Yannic Kilcher — ~30 min. Detailed review of AdamW optimizer mechanics and weight decay effects; essential background for understanding why LR and weight decay interact during LoRA training.
- [Gradient Checkpointing and Packing for Efficient LLM Fine-Tuning](https://www.youtube.com/watch?v=g68qlo9Izf0) — HuggingFace — ~20 min. Practical walkthrough of gradient checkpointing, sequence packing, and per-device batch size tuning to maximize GPU utilization during SFT.

## Blog Posts / Articles

- [LoRA insights from hundreds of experiments](https://lightning.ai/pages/community/lora-insights/) — Sebastian Raschka, Lightning AI. **Required reading this week.** The most comprehensive empirical study of LoRA hyperparameters available publicly.
- [Practical Tips for Finetuning LLMs Using LoRA](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) — Sebastian Raschka. The written companion; re-read sections on rank and target_modules.
- [Hyperparameter Search for LoRA](https://huggingface.co/blog/ray-tune) — HuggingFace blog on using Ray Tune for hyperparameter search with HF Trainer.

## GitHub Repos

- [wandb/wandb](https://github.com/wandb/wandb) — Weights & Biases Python client. The `examples/` directory contains W&B sweep configuration examples; study `sweep.yaml` patterns for hyperparameter grids.
- [optuna/optuna](https://github.com/optuna/optuna) — Bayesian hyperparameter optimization framework. Alternative to W&B sweeps; supports pruning of bad trials mid-run to save compute.
- [huggingface/peft](https://github.com/huggingface/peft) — PEFT library source code. The `examples/` directory contains LoRA training scripts with configurable rank and target modules.

## Documentation

- [SFTConfig parameter reference](https://huggingface.co/docs/trl/sft_trainer#trl.SFTConfig) — Complete list of all SFT hyperparameters and their defaults.
- [EarlyStoppingCallback](https://huggingface.co/docs/transformers/main_classes/callback#transformers.EarlyStoppingCallback) — How to add early stopping to HuggingFace Trainer.

## Optional / Bonus

- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — Kaplan et al. 2020. The original scaling laws; relevant context for understanding when more epochs help vs. more data.
- [Warm-up and Batch Size Interactions](https://arxiv.org/abs/2104.14572) — Academic analysis of LR warmup scheduling; deeper background on warmup_ratio choices.
