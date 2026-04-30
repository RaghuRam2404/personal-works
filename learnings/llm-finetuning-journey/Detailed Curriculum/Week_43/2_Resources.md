# Week 43 Resources

## Papers

- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) — Rafailov et al. 2023. Read fully, including Appendix A.1. The derivation in the appendix is the acceptance criterion for this week.

## Videos

- [Umar Jamil — DPO Direct Preference Optimization Paper Explained](https://www.youtube.com/watch?v=hvGa5Mba4c8) — Umar Jamil, ~50 min. Walks through the derivation with visual aids. Watch after reading the paper.

## Blog Posts / Articles

- [Fine-tuning LLMs with DPO](https://huggingface.co/blog/dpo-trl) — HuggingFace blog. Shows a minimal working DPO example with TRL. Good quick reference after reading the paper.
- [RLHF vs DPO vs GRPO: An Overview](https://huggingface.co/blog/pref_align_and_rlvr) — HuggingFace. Places DPO in context with the broader alignment landscape.
- [Understanding DPO](https://gist.github.com/vllm-project/dpo-understanding) — Technical notes with step-by-step math. Alternative exposition if the paper's notation is unclear.

## GitHub Repos

- [philschmid's DPO notebook](https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/rl-with-llms-in-2025-dpo.ipynb) — The notebook referenced in the assignment. Run end-to-end.
- [TRL DPOTrainer source](https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py) — Read `compute_loss()` to see exactly how log-ratios are computed.
- [Unsloth DPO examples](https://github.com/unslothai/unsloth/blob/main/README.md) — Unsloth's DPO is ~2× faster than vanilla TRL on the same hardware.

## Documentation

- [TRL DPOTrainer docs](https://huggingface.co/docs/trl/dpo_trainer) — Full parameter reference. Pay attention to `beta`, `loss_type` (default is "sigmoid"), and `max_prompt_length`.
- [HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) — The training dataset. Check the dataset card to understand how preference labels were assigned.

## Optional / Bonus

- [A General Theoretical Paradigm to Understand Learning from Human Feedback](https://arxiv.org/abs/2310.12036) — Azar et al. 2023. Theoretical extension of DPO. Read if you want to understand IPO (identity preference optimization), which avoids the Bradley-Terry assumption.
- [RSO: Statistical Rejection Sampling Optimization](https://arxiv.org/abs/2309.06657) — An alternative to DPO that accounts for distributional shift. Relevant when your preference data is stale.
- [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760) — Gao et al. 2022. Important background on reward hacking — shows quantitatively how RM scores and true quality diverge as KL increases.
