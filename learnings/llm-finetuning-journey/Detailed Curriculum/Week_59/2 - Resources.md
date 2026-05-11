# Week 59 Resources

## Papers

- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) — Rafailov et al., 2023. The original DPO paper; read the derivation in the appendix.
- [A General Theoretical Paradigm to Understand Learning from Human Feedback](https://arxiv.org/abs/2310.12036) — Azar et al., 2024. Theoretical framework unifying DPO, PPO, and other preference methods.
- [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691) — Hong et al., 2024. An alternative to DPO without requiring a reference model.
- [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734) — Meng et al., 2024. Simplified DPO variant; compare if DPO instability is a problem.

## Videos

- [Umar Jamil — DPO paper explained](https://www.youtube.com/watch?v=hvGa5Mba4c8) — Umar Jamil, ~50m. Clear mathematical walkthrough of DPO derivation.
- [Yannic Kilcher — DPO](https://www.youtube.com/watch?v=XZLc09hkMwA) — Yannic Kilcher, ~35m.

## Blog Posts / Articles

- [Philschmid — DPO practical guide](https://www.philschmid.de/dpo-align-llms-in-2024-with-trl) — Step-by-step DPO implementation with TRL.
- [Hugging Face — DPO Trainer documentation](https://huggingface.co/docs/trl/dpo_trainer) — Official TRL DPO documentation with all config options.
- [NousResearch — DPO tips and tricks](https://huggingface.co/blog/pref-tuning) — Practical lessons from applying DPO to large models.

## GitHub Repos

- [huggingface/trl — DPOTrainer](https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py) — Source code for DPOTrainer; study the reference model handling with LoRA.
- [philschmid/deep-learning-pytorch-huggingface](https://github.com/philschmid/deep-learning-pytorch-huggingface) — Phil Schmid's DPO notebooks.

## Documentation

- [TRL DPOConfig](https://huggingface.co/docs/trl/dpo_trainer#trl.DPOConfig) — All DPO configuration parameters; critical for beta, max_prompt_length, max_length.
- [Weights & Biases — DPO metrics guide](https://wandb.ai/capecape/huggingface/reports/Direct-Preference-Optimization-Fine-tuning-Mistral-7B-with-DPO--Vmlldzo2NTIyMDU3) — Reference W&B report showing what DPO metrics should look like.

## Optional / Bonus

- [IPO: A General Framework for Preference Optimization](https://arxiv.org/abs/2310.12036) — Identity Preference Optimization; addresses some DPO instabilities.
- [RSO: Statistical Rejection Sampling Optimization](https://arxiv.org/abs/2309.06657) — On-policy preference optimization variant; relevant to your "on-policy pairs" strategy.
