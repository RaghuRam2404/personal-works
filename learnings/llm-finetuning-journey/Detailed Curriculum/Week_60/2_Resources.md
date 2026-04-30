# Week 60 Resources

## Papers

- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) — Shao et al., 2024. The paper introducing GRPO; read Section 3 on the algorithm.
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) — The 2025 paper applying GRPO to reasoning at scale; Section 2 on GRPO is essential.
- [Reinforcement Learning from Human Feedback (RLHF) survey](https://arxiv.org/abs/2307.01852) — Lambert et al., 2023. Comprehensive survey placing GRPO in context.

## Videos

- [Yannic Kilcher — DeepSeek-R1 explained](https://www.youtube.com/watch?v=QNWpnEKiRuE) — Yannic Kilcher, ~45m. Covers GRPO deeply.
- [Andrej Karpathy — LLM Year in Review / RLVR section](https://www.youtube.com/watch?v=Zaf218pEb-8) — Karpathy, segment on GRPO and verifiable rewards.

## Blog Posts / Articles

- [Hugging Face GRPO Trainer documentation](https://huggingface.co/docs/trl/grpo_trainer) — Official docs for GRPOConfig and GRPOTrainer.
- [Zephyr: Direct Distillation of LM Alignment](https://huggingface.co/blog/rlhf) — HuggingFace alignment blog; background on the SFT → DPO → GRPO pipeline.
- [willthompson.io — GRPO from scratch](https://willthompson.io/grpo-explained) — Clear mathematical derivation of GRPO from first principles.

## GitHub Repos

- [huggingface/trl — GRPOTrainer](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py) — Source code for GRPOTrainer.
- [unslothai/unsloth](https://github.com/unslothai/unsloth) — Unsloth GRPO example notebooks for Qwen models.
- [evel-engine/sql-grpo-example](https://github.com/huggingface/trl/tree/main/examples/scripts/grpo.py) — TRL's GRPO example script.

## Documentation

- [TRL GRPOConfig](https://huggingface.co/docs/trl/grpo_trainer#trl.GRPOConfig) — All GRPO configuration parameters.
- [Unsloth — saving and merging guide](https://github.com/unslothai/unsloth/wiki) — How to merge LoRA and push to Hub.
- [psycopg2 pool](https://www.psycopg.org/docs/pool.html) — ThreadedConnectionPool for multi-threaded reward function execution.

## Optional / Bonus

- [REINFORCE algorithm tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html) — PyTorch REINFORCE; GRPO is a variance-reduced extension of REINFORCE.
- [pass@K metric paper (HumanEval)](https://arxiv.org/abs/2107.03374) — How pass@K is computed and interpreted for code generation; directly relevant to SQL.
