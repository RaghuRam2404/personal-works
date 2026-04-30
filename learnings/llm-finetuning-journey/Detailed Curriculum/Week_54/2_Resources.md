# Week 54 Resources

## Papers

- [Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing](https://arxiv.org/abs/2406.08464) — Xu et al., 2024. The core Magpie method.
- [Genie: Achieving Human Parity in Content-Grounded Datasets Generation](https://arxiv.org/abs/2401.14367) — Yehudai et al., 2024. Grounded synthetic data generation.
- [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560) — Wang et al., 2023. The foundational self-instruct method.
- [Alpaca: A Strong, Replicable Instruction-Following Model](https://crfm.stanford.edu/2023/03/13/alpaca.html) — Taori et al., 2023. First major application of self-instruct to LLaMA.
- [WizardLM: Empowering Large Language Models to Follow Complex Instructions](https://arxiv.org/abs/2304.12244) — Xu et al., 2023. Evol-Instruct for increasing difficulty of synthetic data.

## Videos

- [Sebastian Raschka — Instruction fine-tuning and synthetic data](https://www.youtube.com/watch?v=XfpMkf4rD6E) — Sebastian Raschka, ~45m. Practical overview of synthetic data pipelines.
- [Weights and Biases — Logging ML pipelines](https://www.youtube.com/watch?v=krMRk30MJCE) — W&B, ~20m. How to instrument your generation pipeline with W&B.

## Blog Posts / Articles

- [OpenAI API cookbook — async parallelism](https://cookbook.openai.com/examples/api_request_parallel_processor) — Exact pattern for async batch generation with rate limiting.
- [Argilla blog — synthetic data for NLP](https://argilla.io/blog/synthetic-data/) — Practical lessons from large-scale synthetic data projects.
- [Hamel Husain — Generating SQL training data](https://hamel.dev/blog/posts/sql/) — Domain-specific synthetic SQL generation walkthrough.

## GitHub Repos

- [Magpie official code](https://github.com/magpie-align/magpie) — Reference implementation of Magpie alignment data synthesis.
- [openai/openai-cookbook](https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py) — Production-grade async API call pipeline with rate limiting.
- [argilla-io/distilabel](https://github.com/argilla-io/distilabel) — Framework for large-scale synthetic data generation pipelines.
- [gretelai/gretel-synthetics](https://github.com/gretelai/gretel-synthetics) — Synthetic tabular data; useful for schema generation diversity.

## Documentation

- [OpenAI Python async client](https://github.com/openai/openai-python#async-usage) — Official async client documentation.
- [Anthropic API rate limits](https://docs.anthropic.com/en/api/rate-limits) — Know your limits before designing concurrency.
- [psycopg2 documentation](https://www.psycopg.org/docs/) — PostgreSQL Python driver; use for execution validation.
- [TimescaleDB SQL reference](https://docs.timescale.com/api/latest/) — Ground truth for all TimescaleDB function signatures.

## Optional / Bonus

- [Evol-Instruct paper](https://arxiv.org/abs/2304.12244) — Method for progressively increasing instruction complexity; useful for generating Expert-level examples.
- [DEITA paper](https://arxiv.org/abs/2312.15685) — Automated complexity and quality scoring of generated data; could replace manual difficulty specification.
- [sqlglot documentation](https://sqlglot.com/sqlglot.html) — Use for dialect conversion (MySQL → PostgreSQL) and AST-based complexity scoring.
