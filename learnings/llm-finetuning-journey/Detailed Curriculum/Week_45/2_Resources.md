# Week 45 Resources

## Documentation

- [Unsloth DPO training guide](https://github.com/unslothai/unsloth/blob/main/README.md#dpo-support) — Unsloth's DPO training documentation. Includes memory requirements for 7B models and VRAM estimates.
- [TRL DPOTrainer docs](https://huggingface.co/docs/trl/dpo_trainer) — Full parameter reference. Check `length_normalization`, `loss_type`, and `label_smoothing` options.
- [Unsloth Colab notebooks](https://github.com/unslothai/unsloth#-finetune-for-free) — Pre-configured notebooks for DPO training on various model sizes.

## Blog Posts / Articles

- [Fine-tune your own LLM with DPO using TRL](https://huggingface.co/blog/dpo-trl) — HuggingFace. Step-by-step walkthrough with the same API you are using.
- [DPO implementation tips from TRL developers](https://huggingface.co/docs/trl/main/en/dpo_trainer#tips-and-common-tricks) — Practical advice on β tuning, learning rate, and common failure modes.
- [The TRL Forum (DPO issues)](https://huggingface.co/spaces/trl-lib/model-card) — If your DPO loss goes negative or reward_margin does not improve, search the TRL GitHub issues for your specific error message.

## GitHub Repos

- [TRL source — DPOTrainer.compute_loss()](https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py) — Read `compute_loss()` to trace exactly how log-ratios and the loss are computed for your specific TRL version.

## Papers

- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) — Rafailov et al. 2023. Reference for this week's task; especially Appendix B for practical training details.

## Videos

- [Umar Jamil — DPO: Direct Preference Optimization](https://www.youtube.com/watch?v=hvGa5Mba4c8) — Umar Jamil — ~1h. Full derivation of the DPO loss from first principles plus a PyTorch implementation walkthrough; the best technical video on DPO available. Watch before starting your training run.
- [Trelis Research — DPO Fine-Tuning Tutorial](https://www.youtube.com/watch?v=UhoPL5Btj7I) — Trelis Research — ~35 min. Practical guide to DPO training with TRL's DPOTrainer; covers dataset format, beta tuning, and reward margin monitoring.
- [HuggingFace — Fine-tune with DPO (SMOL Talk)](https://www.youtube.com/watch?v=6A0RKNbPdLY) — HuggingFace — ~20 min. Walkthrough of the philschmid DPO notebook that served as the reference for this week; shows exactly how to configure DPOTrainer for a domain model.

## Optional / Bonus

- [Towards the Fundamental Limits of Knowledge over Finite Domains](https://arxiv.org/abs/2310.07019) — Theoretical analysis of when DPO can fail; relevant if your reward_margin does not improve.
- [SPIN: Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models](https://arxiv.org/abs/2401.01335) — An alternative to DPO for iteratively improving a model by having it compete against its own previous version. Could be used in Week 50 iteration if DPO plateaus.
- [Spider 2.0 — Enterprise Text-to-SQL Benchmark](https://spider2-sql.github.io/) — The next-generation SQL benchmark with enterprise-level complexity. Use it as a source of hard test queries for your eval if it covers PostgreSQL schemas.
