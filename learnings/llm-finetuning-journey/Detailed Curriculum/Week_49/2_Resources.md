# Week 49 Resources

## Papers (Required Skim)

- [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306) — Ethayarajh et al. 2024. Read: abstract, Section 2 (the loss), Table 1 (comparison with DPO). Skip the full experiments.
- [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691) — Hong et al. 2024. Read: abstract, Section 3 (the ORPO loss), Section 5 (comparison table). Skip RLHF background section.
- [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734) — Meng et al. 2024. Read: abstract, Section 3 (SimPO formulation), Table 1 (AlpacaEval results). Skim experiments.

## Blog Posts / Articles

- [A Survey of Preference-Based Reinforcement Learning Methods](https://huggingface.co/blog/pref_align_and_rlvr) — HuggingFace blog. The most comprehensive recent survey. Places all 6 methods in context.
- [DPO vs KTO vs SimPO: Practical Comparison](https://kaitchup.substack.com/p/dpo-kto-simpo-orpo) — The Kaitchup. Applied perspective on which method to choose in practice.

## GitHub Repos / Documentation

- [TRL KTOTrainer docs](https://huggingface.co/docs/trl/kto_trainer) — Full API reference for KTO.
- [TRL ORPOTrainer docs](https://huggingface.co/docs/trl/orpo_trainer) — Full API reference for ORPO.
- [TRL SimPOTrainer docs](https://huggingface.co/docs/trl/simpo_trainer) — Full API reference for SimPO.
- [TRL DPO variants overview](https://huggingface.co/docs/trl/dpo_trainer#loss-types) — Shows that TRL's DPOTrainer supports multiple loss types (standard, sigmoid, IPO, etc.) via the `loss_type` parameter.

## Videos

- [Yannic Kilcher — KTO: Model Alignment as Prospect Theoretic Optimization](https://www.youtube.com/watch?v=Q8CHqpCFHmk) — Yannic Kilcher — ~30 min. Paper walkthrough of KTO including the Kahneman-Tversky utility framing and why unpaired feedback works; watch before running your KTO experiment.
- [AI Coffee Break — ORPO: Monolithic Preference Optimization](https://www.youtube.com/watch?v=nNZRKdwbzCQ) — AI Coffee Break with Letitia — ~20 min. Clear explanation of why ORPO eliminates the reference model and how the combined SFT+preference loss works in practice.
- [Trelis Research — DPO vs KTO vs ORPO Comparison](https://www.youtube.com/watch?v=UhoPL5Btj7I) — Trelis Research — ~30 min. Practical comparison of alignment methods with TRL; useful for understanding the implementation differences before running all three on your SQL dataset.

## Optional / Bonus

- [From r to Q*: Your Language Model is Secretly a Q-Function](https://arxiv.org/abs/2404.12358) — Theoretical unification of DPO, KTO, and reward-free methods. Advanced reading for those interested in the theory.
- [SPIN: Self-Play Fine-Tuning](https://arxiv.org/abs/2401.01335) — An alternative paradigm: instead of preference data, the model plays against its own previous version. Good alternative for when no preference data is available.
- [Alignment handbook](https://github.com/huggingface/alignment-handbook) — HuggingFace's collection of alignment training recipes. Contains configs for DPO, KTO, ORPO, and SFT across many model families. Useful reference for practical hyperparameter choices.
