# Week 48 Resources

## Documentation

- [Unsloth GRPO training guide](https://github.com/unslothai/unsloth) — Memory-optimized GRPO. Check the README section on GRPO for the exact API with the current Unsloth version.
- [TRL GRPOConfig reference](https://huggingface.co/docs/trl/grpo_trainer#trl.GRPOConfig) — All hyperparameters. Crucial for understanding `num_generations`, `temperature`, `kl_coef`.
- [RunPod documentation](https://docs.runpod.io/) — GPU instance setup, SSH access, volume persistence. Read before starting the pod.
- [W&B GRPO metrics](https://docs.wandb.ai/guides/integrations/huggingface) — How TRL reports to W&B; which metrics to monitor.

## Blog Posts / Articles

- [Training DeepSeek-R1 yourself: practical guide](https://huggingface.co/blog/open-r1) — HuggingFace Open-R1 blog. The practical walkthrough most similar to what you are doing. Read before starting the run.
- [GRPO in Practice: What Works and What Doesn't](https://www.interconnects.ai/p/grpo-in-practice) — Nathan Lambert. Lessons from running GRPO at various scales.

## Papers

- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning](https://arxiv.org/abs/2402.03300) — Reference for GRPO algorithm details (Section 3). Re-read if any training behavior is puzzling.
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL](https://arxiv.org/abs/2501.12948) — The gold standard for what GRPO can produce with a verifiable reward.

## GitHub Repos

- [Open-R1](https://github.com/huggingface/open-r1) — HuggingFace's open replication of DeepSeek-R1 training pipeline. Study `src/open_r1/grpo.py` for a production GRPO training script.
- [Unsloth notebooks](https://github.com/unslothai/unsloth#-finetune-for-free) — Pre-configured RunPod/Colab notebooks for GRPO training.
- [Open-Reasoner](https://github.com/openreasoner/openreasoner) — Another open RLVR training framework. Review for alternative GRPO configurations.

## Videos

- [Daniel Han — Unsloth GRPO Tutorial](https://www.youtube.com/watch?v=aQmoog_s8_k) — Unsloth AI — ~20 min. Covers the Unsloth GRPO API, memory footprint on A100, and how to hook a custom reward function into the GRPOTrainer; the closest walkthrough to what you are running this week.
- [Trelis Research — GRPO Fine-Tuning Walkthrough](https://www.youtube.com/watch?v=MtlHhNn2RDYQ) — Trelis Research — ~25 min. End-to-end GRPO training run with monitoring; covers W&B metrics to watch (reward_std, kl_div, entropy) during the live A100 session.
- [Yannic Kilcher — DeepSeek-R1 Paper Explained](https://www.youtube.com/watch?v=SKID7bqnIvU) — Yannic Kilcher — ~45 min. The GRPO training details in Section 3 are directly applicable to diagnosing training behavior during your run; rewatch the advantage computation section if reward_std collapses.

## Optional / Bonus

- [DAPO: An Open-Source LLM RL System at Scale](https://arxiv.org/abs/2503.14476) — Byte Dance's production GRPO training system. The "clip-higher" trick and entropy coefficient adjustment are relevant if your reward_std collapses.
- [The Arithmetic of Compute for RLVR](https://rentry.org/RLVR-compute) — Community analysis of compute requirements for GRPO training at different scales. Use for planning your Week 50 iteration budget.
