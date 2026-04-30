# Week 42 Resources

## Papers

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — Schulman et al. 2017. The original PPO paper. Read sections 1–3 and the experiments. The clipping objective is in Equation 7.
- [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) — Christiano et al. 2017. The foundational RLHF paper. Shows that human preference signals can be used to train reward models for RL.
- [Training Language Models to Follow Instructions with Human Feedback (InstructGPT)](https://arxiv.org/abs/2203.02155) — Ouyang et al. 2022. Read sections 3 and 4 fully. Section 3 gives the three-stage pipeline; Section 4 gives the key finding that 1.3B RLHF model outperforms 175B GPT-3.

## Videos

- [Yannic Kilcher — InstructGPT / ChatGPT Explained](https://www.youtube.com/watch?v=VPRSBzXzavo) — Yannic Kilcher, ~30 min. Clear walkthrough of the three stages with commentary on the results.
- [John Schulman — OpenAI PPO](https://www.youtube.com/watch?v=5P7I-xPq8u8) — OpenAI, ~45 min. Schulman's own lecture on PPO. Useful for intuition behind the clip.

## Blog Posts / Articles

- [The 37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) — Shengyi Costa Huang. Critically important for anyone implementing PPO. The 13 LLM-specific details at the bottom are especially relevant.
- [Spinning Up — PPO documentation](https://spinningup.openai.com/en/latest/algorithms/ppo.html) — OpenAI. Read the GAE section. Has pseudocode for the full algorithm.
- [RLHF: From Zero to ChatGPT](https://huyenchip.com/2023/05/02/rl-human-feedback-rlhf.html) — Chip Huyen. High-level narrative of the RLHF landscape. Good complement to the technical paper.

## GitHub Repos

- [TRL PPOTrainer source](https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py) — HuggingFace TRL. The file you are annotating this week. Use `git blame` to find when specific lines were added; the commit messages often explain the design choices.
- [CleanRL PPO implementation](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py) — Single-file, readable PPO for continuous/discrete control. Read this before the TRL version to understand the core logic without LLM-specific complexity.

## Documentation

- [TRL PPOTrainer docs](https://huggingface.co/docs/trl/ppo_trainer) — HuggingFace. The official documentation with configuration parameters and usage examples.
- [TRL PPOConfig reference](https://huggingface.co/docs/trl/main/en/ppo_trainer#trl.PPOConfig) — All hyperparameters with defaults. Check `kl_coef`, `cliprange`, `vf_coef`.

## Optional / Bonus

- [TRPO paper](https://arxiv.org/abs/1502.05477) — Schulman et al. 2015. The predecessor to PPO using second-order optimization. Understanding TRPO makes PPO's motivation clearer.
- [Secrets of RLHF in Large Language Models Part I](https://arxiv.org/abs/2307.04964) — Zheng et al. 2023. Practical lessons from training large models with RLHF. Good for understanding the gap between the paper recipe and production.
