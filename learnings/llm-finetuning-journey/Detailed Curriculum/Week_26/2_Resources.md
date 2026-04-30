# Week 26 Resources — Domain Dataset Construction

## Papers

- [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560) — Wang et al. 2022. Required reference for this week's Self-Instruct generation.
- [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) — Zhou et al. 2023. Demonstrates quality > quantity for fine-tuning datasets; motivates your hand-written Tier 2.
- [Spider 2.0: Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows](https://arxiv.org/abs/2411.07763) — Spider 2.0, 2024. The next-generation text-to-SQL benchmark that uses real enterprise databases including PostgreSQL.
- [DAIL-SQL: Efficient Prompt Engineering for Large Language Models on Text-to-SQL](https://arxiv.org/abs/2308.15363) — Gao et al. 2023. State-of-the-art prompt engineering for text-to-SQL; useful for understanding how to structure your training examples.

## Videos

- [Building fine-tuning datasets with synthetic data](https://www.youtube.com/watch?v=s9I3BMRW4go) — Maxime Labonne (~45 min). Practical guide to synthetic dataset construction with quality filtering.
- [TimescaleDB tutorial for developers](https://www.youtube.com/watch?v=dYfLBX0bnrY) — Timescale, (~30 min). Covers time_bucket, continuous aggregates, and hypertables — the functions you need for your hand-written examples.

## Blog Posts / Articles

- [TimescaleDB documentation: time_bucket](https://docs.timescale.com/api/latest/hyperfunctions/time_bucket/) — Official TimescaleDB API docs. Reference for all time_bucket variants.
- [TimescaleDB documentation: Continuous Aggregates](https://docs.timescale.com/use-timescale/latest/continuous-aggregates/) — Official docs for continuous aggregates; essential for Tier 2 examples.
- [PostgreSQL documentation: Window Functions](https://www.postgresql.org/docs/current/tutorial-window.html) — Official PostgreSQL docs. Reference for LAG, LEAD, RANK, NTILE examples.
- [PostgreSQL documentation: JSONB](https://www.postgresql.org/docs/current/datatype-json.html) — Reference for JSONB operators used in your examples.
- [Maxime Labonne's synthetic data guide](https://mlabonne.github.io/blog/posts/2024-05-15-Generate_Synthetic_Data.html) — Comprehensive guide to generating fine-tuning data with LLMs.

## GitHub Repos

- [tobymao/sqlglot](https://github.com/tobymao/sqlglot) — SQL parser with PostgreSQL dialect support.
- [taoyds/spider](https://github.com/taoyds/spider) — Spider dataset; includes train/dev/test splits and schema metadata.
- [bird-bench/mini-dev](https://github.com/bird-bench/mini-dev) — BIRD mini-dev set for rapid iteration.
- [huggingface/datasets](https://github.com/huggingface/datasets) — HuggingFace datasets library; `push_to_hub` documentation.

## Documentation

- [psycopg2 documentation](https://www.psycopg.org/docs/) — PostgreSQL Python adapter for SQL execution verification.
- [Docker PostgreSQL official image](https://hub.docker.com/_/postgres) — How to run PostgreSQL 16 locally for SQL verification.
- [HuggingFace Hub dataset upload guide](https://huggingface.co/docs/datasets/upload_dataset) — How to create, document, and publish your dataset.

## Optional / Bonus

- [T-SQL vs. PostgreSQL cheat sheet](https://www.sqlines.com/postgresql-to-sql-server) — SQLines reference for understanding SQL dialect differences; useful for extending beyond Spider/BIRD.
- [DINOv2 data curation paper](https://arxiv.org/abs/2309.10500) — Meta's approach to data curation for vision models; principles transfer to NLP dataset quality control.
- [Evol-Instruct: WizardLM method](https://arxiv.org/abs/2304.12244) — How to automatically increase instruction complexity in Self-Instruct; useful for generating harder SQL examples in v2.
