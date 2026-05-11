# Week 62 Resources

## Papers

- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena (Zheng et al. 2023)](https://arxiv.org/abs/2306.05685) — Discusses LLM evaluation reliability; relevant for justifying execution-based over judge-based eval.

## Videos

- [AI Explained — GPT-4o vs specialized models](https://www.youtube.com/watch?v=U_pxVWPpNjQ) — 20m overview of generalist vs specialist model comparisons.

## Blog Posts / Articles

- [SQLCoder: Defog’s SQL-Specialized LLM](https://defog.ai/blog/open-sourcing-sqlcoder2-7b) — Defog blog covering SQLCoder’s methodology and benchmark scores; your primary competitor reference.
- [Weights & Biases — ML experiment comparison best practices](https://wandb.ai/authors/ml-experiments-best-practices/reports/Best-Practices-for-ML-Experiments--VmlldzozODgyMTU2) — How to design fair model comparisons.
- [mlxtend — McNemar's test for paired model evaluation (Sebastian Raschka)](https://rasbt.github.io/mlxtend/user_guide/evaluate/mcnemar/) — Practical guide to McNemar's test for model comparison with Python implementation examples.
- [Defog SQL benchmark leaderboard](https://github.com/defog-ai/sql-eval#sql-eval-results) — Current published numbers for SQLCoder and other models.

## GitHub Repos

- [defog-ai/sql-eval](https://github.com/defog-ai/sql-eval) — Defog's eval framework; use for standardized Defog benchmark scores.
- [openai/openai-python](https://github.com/openai/openai-python) — OpenAI Python client; use for GPT-4o API calls.
- [anthropics/anthropic-sdk-python](https://github.com/anthropics/anthropic-sdk-python) — Anthropic Python client for Claude.

## Documentation

- [OpenAI API pricing](https://openai.com/api/pricing/) — Current GPT-4o pricing for cost analysis.
- [Anthropic API pricing](https://www.anthropic.com/api) — Claude 3.5 Sonnet pricing.
- [Together AI model catalog](https://api.together.xyz/models) — For running DeepSeek-Coder-V2-Lite via API if local VRAM is insufficient.
- [DeepSeek-Coder-V2-Lite model card](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct) — Official model card with prompt template.

## Optional / Bonus

- [statsmodels.stats.contingency_tables.mcnemar documentation](https://www.statsmodels.org/dev/generated/statsmodels.stats.contingency_tables.mcnemar.html) — Statistical implementation for the McNemar's test (note: McNemar lives in statsmodels, not scipy.stats).
- [Chatbot Arena methodology paper](https://arxiv.org/abs/2403.04132) — How to compare models at scale; useful context for evaluation design.
- [EleutherAI LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) — Large-scale evaluation framework; overkill for your use case but useful to understand.
