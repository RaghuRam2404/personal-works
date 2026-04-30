# Week 46 Resources

## Papers

- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) — Shao et al. 2024. Introduces GRPO. Read Section 3 ("Reinforcement Learning from Feedback") fully. The GRPO algorithm is in Algorithm 1.
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) — DeepSeek AI 2025. Read this fully. Sections 2 and 3 describe the cold start + GRPO pipeline. Section 4 has the emergent reasoning behavior ("aha moment") analysis.

## Videos

- [Andrej Karpathy — LLM Year in Review 2025](https://www.youtube.com/watch?v=Zaf218pEb-8) — Andrej Karpathy — ~30 min. Watch the RLVR section specifically (~15 min). Karpathy's intuitive explanation of why verifiable rewards are transformative.
- [Yannic Kilcher — DeepSeek-R1 Paper Explained](https://www.youtube.com/watch?v=SKID7bqnIvU) — Yannic Kilcher — ~45 min. Clear technical walkthrough of the R1 paper, the GRPO algorithm, and the emergent chain-of-thought behavior. Watch after reading the paper.
- [Umar Jamil — GRPO from Scratch in PyTorch](https://www.youtube.com/watch?v=G5cpnTl6Dew) — Umar Jamil — ~55 min. Step-by-step mathematical derivation and PyTorch implementation of GRPO, with code walkthrough.

## Blog Posts / Articles

- [HuggingFace LLM Course Chapter 12: Implementing GRPO](https://huggingface.co/learn/llm-course/chapter12/4) — HuggingFace. Practical walkthrough of GRPO with TRL. Read fully before annotating the GRPOTrainer.
- [GRPO: A Simple Alternative to PPO](https://huggingface.co/blog/grpo) — HuggingFace blog. Concise overview with code examples.
- [Understanding DeepSeek-R1 and its Surprising Emergent Reasoning](https://www.interconnects.ai/p/reasoning-models-from-rl) — Nathan Lambert. Analysis of why RLVR produces reasoning chains.

## GitHub Repos

- [TRL GRPOTrainer source](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py) — The file you are annotating. Focus on `_generate_completions`, `compute_advantages`, and `compute_loss`.
- [TRL GRPO examples](https://github.com/huggingface/trl/tree/main/examples/scripts) — Look for any `grpo_` prefixed scripts. These show end-to-end usage.
- [Unsloth GRPO](https://github.com/unslothai/unsloth) — Unsloth's GRPO implementation, optimized for 5GB VRAM. This is what you will use in Week 48.
- [Open-Reasoner-Zero](https://github.com/openreasoner/openreasoner) — Open-source replication of DeepSeek-R1's RLVR training. Study for reference.

## Documentation

- [TRL GRPOConfig documentation](https://huggingface.co/docs/trl/grpo_trainer) — All hyperparameters. Pay attention to `num_generations`, `max_completion_length`, and `reward_model` vs `reward_fn` arguments.

## Optional / Bonus

- [STILL-2: Improving LLM Reasoning with Process Rewards and GRPO](https://arxiv.org/abs/2501.12599) — Process reward models combined with GRPO. Relevant if you want to reward intermediate reasoning steps in SQL.
- [DAPO: An Open-Source LLM Reinforcement Learning System](https://arxiv.org/abs/2503.14476) — Byte Dance's open-source GRPO improvements. Includes clip-higher trick and entropy regularization that can help when training plateaus.
