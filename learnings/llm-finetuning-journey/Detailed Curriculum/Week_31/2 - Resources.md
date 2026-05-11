# Week 31 Resources

## Papers

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al. 2021. Re-read Section 7 (ablations on target modules and rank).
- [Parameter-Efficient Fine-Tuning Methods Survey](https://arxiv.org/abs/2403.14608) — Comprehensive comparison of LoRA and alternatives; skim Table 1 for a quick overview.

## Videos

- [Practical Tips for Fine-Tuning LLMs with LoRA](https://www.youtube.com/watch?v=XpoKB3usmKc) — Sebastian Raschka — ~35 min. Empirical guidance on target_modules selection and rank choice; data from hundreds of experiments directly relevant to this week's sweep.
- [LoRA Fine-Tuning Explained with the PEFT Library](https://www.youtube.com/watch?v=eC6Hd1hFvos) — Sam Witteveen — ~30 min. Step-by-step tutorial using HuggingFace PEFT to apply LoRA to a 7B model; covers LoraConfig parameters and merging adapters.
- [PEFT and LoRA Walkthrough with HuggingFace](https://www.youtube.com/watch?v=YVU5wAA6Txo) — HuggingFace — ~20 min. Official walkthrough of the PEFT library API, including how to set target_modules, save adapters, and push to Hub.

## Blog Posts / Articles

- [Practical Tips for Finetuning LLMs Using LoRA (written)](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) — Sebastian Raschka. **Required reading this week.** Contains empirical data on which target_modules to use and rank ablations.
- [LoRA insights from 1000+ experiments](https://lightning.ai/pages/community/lora-insights/) — Lightning AI / Raschka. Complementary to the article above.

## GitHub Repos

- [peft library](https://github.com/huggingface/peft) — HuggingFace. Source code; look at `peft/tuners/lora/layer.py` to see how peft implements what you built in Week 30.
- [peft examples](https://github.com/huggingface/peft/tree/main/examples) — Minimal scripts for various tasks and model types.

## Documentation

- [PEFT LoRA documentation](https://huggingface.co/docs/peft/package_reference/lora) — Complete LoraConfig parameter reference.
- [PEFT model saving and loading](https://huggingface.co/docs/peft/tutorial/peft_model_config) — How to save, load, and push adapters to Hub.

## Optional / Bonus

- [LoRA Land: Fine-Tuned Open-Source LLMs that Outperform GPT-4](https://arxiv.org/abs/2405.00732) — Predibase 2024. A systematic study of LoRA fine-tuned models across 25 tasks; useful for understanding what performance levels are achievable with LoRA.
- [Asymmetry in Low-Rank Adapters of Foundation Models](https://arxiv.org/abs/2402.16842) — 2024. Analysis of lora_A vs. lora_B learning dynamics; more advanced but relevant context for understanding rank sweep results.
