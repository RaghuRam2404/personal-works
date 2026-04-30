# Week 62 Resources

## Papers

- [Is Your LLM Secretly a World Model? On the Limits of LLM-as-Judge](https://arxiv.org/abs/2402.17232) — Discusses LLM evaluation reliability; relevant for justifying execution-based over judge-based eval.

## Videos

- [AI Explained — GPT-4o vs specialized models](https://www.youtube.com/watch?v=U_pxVWPpNjQ) — 20m overview of generalist vs specialist model comparisons.

## Blog Posts / Articles

- [SQLCoder: Defog’s SQL-Specialized LLM](https://defog.ai/blog/sqlcoder-7b/) — Defog blog covering SQLCoder’s methodology and benchmark scores; your primary competitor reference.
- [Weights & Biases — ML experiment comparison best practices](https://wandb.ai/authors/ml-experiments-best-practices/reports/Best-Practices-for-ML-Experiments--VmlldzozODgyMTU2) — How to design fair model comparisons.
- [Eric Ma — McNemar's test for ML](https://ericmjl.github.io/blog/2019/2/12/mcnemars-test-for-evaluating-model-pairs/) — Practical guide to McNemar's test for model comparison.
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

- [scipy.stats.mcnemar documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.contingency.mcnemar.html) — Statistical implementation for the McNemar's test.
- [Chatbot Arena methodology paper](https://arxiv.org/abs/2403.04132) — How to compare models at scale; useful context for evaluation design.
- [EleutherAI LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) — Large-scale evaluation framework; overkill for your use case but useful to understand.
