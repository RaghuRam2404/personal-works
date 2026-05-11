# Week 50 Resources

## Blog Posts / Articles

- [A Recipe for Training Neural Networks](http://karpathy.github.io/2019/04/25/recipe/) — Andrej Karpathy. The definitive guide to debugging and iterating on neural network training. Read section "The training loop" and "Diagnose which part is broken." Directly applicable to your GRPO iteration.
- [Debugging Machine Learning Models](https://developers.google.com/machine-learning/testing-debugging) — Google ML Developers. Systematic framework for ML debugging — checklists for data, model, and training issues.
- [The Bitter Lesson: Scaling vs. Algorithmic Improvements](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) — Rich Sutton. A reminder that sometimes the right answer is "just get more data" rather than algorithmic tricks.

## Papers

- [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476) — Byte Dance 2025. Includes the "clip-higher" trick and entropy regularization for addressing mode collapse during GRPO iteration. Relevant for Failure Mode 3 (reward_std collapse).
- [Deep Reinforcement Learning Doesn't Work Yet](https://www.alexirpan.com/2018/02/14/rl-hard.html) — Alex Irpan. A sobering but accurate analysis of RL failure modes. The section on reward shaping is directly relevant to your SQL reward debugging.

## GitHub Repos

- [Open-R1 training scripts](https://github.com/huggingface/open-r1/tree/main/src/open_r1) — Production GRPO training scripts from HuggingFace. Study the reward function and training loop for reference implementations.
- [TRL GRPO examples](https://github.com/huggingface/trl/tree/main/examples/scripts) — Look for grpo example scripts with custom reward functions.

## Documentation

- [W&B reports for comparing runs](https://docs.wandb.ai/guides/reports) — How to create a W&B report comparing v3 and v3-iter1 side by side. Useful for documenting your iteration log.

## Videos

- [Andrej Karpathy — A Recipe for Training Neural Networks (talk)](https://www.youtube.com/watch?v=yd3AuXKO0dA) — Andrej Karpathy — ~20 min. The video companion to the blog post; covers the systematic debugging approach (overfit one batch first, then scale) directly applicable to your GRPO iteration.
- [Sebastian Raschka — Diagnosing and Fixing Training Issues](https://www.youtube.com/watch?v=iHrBQ4F6Gvs) — Sebastian Raschka — ~30 min. Covers how to read loss curves, identify reward collapse, and decide whether to adjust the reward function or the LoRA configuration.
- [Yannic Kilcher — Deep Reinforcement Learning Doesn't Work Yet](https://www.youtube.com/watch?v=tm1pLDs5KQU) — Yannic Kilcher — ~25 min. Walkthrough of Alex Irpan's classic analysis of RL failure modes; builds intuition for categorizing which failure mode your GRPO run is exhibiting.

## Optional / Bonus

- [Reward Hacking in Reinforcement Learning](https://openai.com/research/faulty-reward-functions) — OpenAI. Classic examples of reward hacking across many domains. Useful for generating new hypotheses about what your SQL reward function might be exploiting.
- [SQLGlot](https://github.com/tobymao/sqlglot) — A Python SQL parser and transpiler. Useful for implementing syntactic validation in your reward function (checking for GROUP BY when aggregation is expected, verifying table references are in the schema, etc.).
