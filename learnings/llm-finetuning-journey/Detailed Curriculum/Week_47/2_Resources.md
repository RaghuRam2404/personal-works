# Week 47 Resources

## Papers

- [Reward Model Ensembles Help Mitigate Overoptimization](https://arxiv.org/abs/2310.02743) — Coste et al. 2023. Shows how reward hacking scales with KL divergence. Relevant for understanding why anti-hack guards are necessary.
- [Let's Reinforce Step by Step](https://arxiv.org/abs/2312.05862) — Process reward models (PRMs) vs. outcome reward models (ORMs). Background for whether to reward intermediate reasoning steps.
- [Scaling LLM Test-Time Compute Optimally Improves Reasoning](https://arxiv.org/abs/2408.03314) — Google DeepMind. Discusses reward shaping for code and math that is analogous to your SQL reward design.

## Blog Posts / Articles

- [Reward Hacking in RLHF: How to Detect and Prevent It](https://huggingface.co/blog/reward-hacking) — HuggingFace. Practical guide to identifying reward hacking in LLM training. Read before finalizing your reward function.
- [Designing Reward Functions for Code Generation](https://www.swebench.com/analysis) — Analysis of reward design for code, analogous to SQL. Focus on the "execution-only vs semantic" tradeoff.

## GitHub Repos

- [TRL GRPOTrainer reward_fn interface](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py) — Read the docstring for `reward_fn` to understand exactly what signature is expected.
- [Unsloth GRPO examples](https://github.com/unslothai/unsloth) — Look for GRPO notebooks showing how to connect a custom reward function.
- [Open-R1 SQL examples](https://github.com/huggingface/open-r1) — HuggingFace's open replication of DeepSeek-R1 training. Check the reward function implementations for code tasks (analogous to SQL).

## Documentation

- [psycopg2 error codes](https://www.psycopg.org/docs/errors.html) — Classification of Postgres error types. Use this to distinguish syntax errors from runtime errors in your reward function.
- [PostgreSQL statement_timeout](https://www.postgresql.org/docs/current/runtime-config-client.html#GUC-STATEMENT-TIMEOUT) — How to set per-query timeouts.

## Videos

- [Yannic Kilcher — DeepSeek-R1 Paper Explained](https://www.youtube.com/watch?v=SKID7bqnIvU) — Yannic Kilcher — ~45 min. Covers the GRPO algorithm, verifiable reward design, and why executable rewards produce emergent reasoning; directly relevant to your SQL reward function design.
- [AI Coffee Break — GRPO Explained](https://www.youtube.com/watch?v=EKZrpMtqRVM) — AI Coffee Break with Letitia — ~20 min. Clear explanation of group-relative advantage estimation and how GRPO differs from PPO in the context of verifiable rewards.
- [Nathan Lambert — RLVR and Reward Hacking in Practice](https://www.youtube.com/watch?v=Zaf218pEb-8) — Interconnects / Nathan Lambert — ~30 min. Practical discussion of reward function failure modes (hacking, mode collapse) and mitigation strategies; useful calibration before finalizing your SQL reward.

## Optional / Bonus

- [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476) — Byte Dance's improvements to GRPO for math. The "clip-higher" trick is particularly relevant for SQL where many completions score 0.
- [Verifiable Reward Scaling Laws](https://arxiv.org/abs/2503.09928) — Empirical study of how GRPO performance scales with K, group size, and reward quality. Useful calibration for your reward design choices.
