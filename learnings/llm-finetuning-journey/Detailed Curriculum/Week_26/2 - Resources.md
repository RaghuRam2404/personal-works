# Week 26 Resources — Domain Dataset Construction

## Papers

- [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560) — Wang et al. 2022. Required reference for this week's Self-Instruct generation.
- [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) — Zhou et al. 2023. Demonstrates quality > quantity for fine-tuning datasets; motivates your hand-written Tier 2.
- [Spider 2.0: Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows](https://arxiv.org/abs/2411.07763) — Spider 2.0, 2024. The next-generation text-to-SQL benchmark that uses real enterprise databases including PostgreSQL.
- [Text-to-SQL Empowered by Large Language Models: A Benchmark Evaluation (DAIL-SQL)](https://arxiv.org/abs/2308.15363) — Gao et al. 2023. State-of-the-art prompt engineering for text-to-SQL; useful for understanding how to structure your training examples.

## Videos

- [Building fine-tuning datasets with synthetic data](https://www.youtube.com/watch?v=s9I3BMRW4go) — Maxime Labonne (~45 min). Practical guide to synthetic dataset construction with quality filtering.
- [TimescaleDB tutorial for developers](https://www.youtube.com/watch?v=dYfLBX0bnrY) — Timescale, (~30 min). Covers time_bucket, continuous aggregates, and hypertables — the functions you need for your hand-written examples.

## Blog Posts / Articles

- [TimescaleDB documentation: time_bucket](https://www.tigerdata.com/docs//api/latest/hyperfunctions/time_bucket/) — Official TimescaleDB API docs. Reference for all time_bucket variants.
- [TimescaleDB documentation: Continuous Aggregates](https://www.tigerdata.com/docs//use-timescale/latest/continuous-aggregates/) — Official docs for continuous aggregates; essential for Tier 2 examples.
- [PostgreSQL documentation: Window Functions](https://www.postgresql.org/docs/current/tutorial-window.html) — Official PostgreSQL docs. Reference for LAG, LEAD, RANK, NTILE examples.
- [PostgreSQL documentation: JSONB](https://www.postgresql.org/docs/current/datatype-json.html) — Reference for JSONB operators used in your examples.
- [The Rise of Agentic Data Generation](https://mlabonne.github.io/blog/posts/2024-07-15_The_Rise_of_Agentic_Data_Generation.html) — Maxime Labonne, July 2024. Combines AgentInstruct and Arena Learning into a synthetic-data pipeline; the agentic refinement loop maps directly onto your Tier 3 SQL generation + Tier 2 hand-written cross-checking design.

## GitHub Repos

- [tobymao/sqlglot](https://github.com/tobymao/sqlglot) — SQL parser with PostgreSQL dialect support.
- [taoyds/spider](https://github.com/taoyds/spider) — Spider dataset; includes train/dev/test splits and schema metadata.
- [bird-bench/mini_dev](https://github.com/bird-bench/mini_dev) — BIRD mini-dev set for rapid iteration.
- [huggingface/datasets](https://github.com/huggingface/datasets) — HuggingFace datasets library; `push_to_hub` documentation.

## Documentation

- [psycopg2 documentation](https://www.psycopg.org/docs/) — PostgreSQL Python adapter for SQL execution verification.
- [Docker PostgreSQL official image](https://hub.docker.com/_/postgres) — How to run PostgreSQL 16 locally for SQL verification.
- [HuggingFace Hub dataset upload guide](https://huggingface.co/docs/datasets/upload_dataset) — How to create, document, and publish your dataset.

## Optional / Bonus

- [T-SQL vs. PostgreSQL cheat sheet](https://www.sqlines.com/postgresql-to-sql-server) — SQLines reference for understanding SQL dialect differences; useful for extending beyond Spider/BIRD.
- [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) — Meta's approach to data curation for vision models; principles transfer to NLP dataset quality control.
- [Evol-Instruct: WizardLM method](https://arxiv.org/abs/2304.12244) — How to automatically increase instruction complexity in Self-Instruct; useful for generating harder SQL examples in v2.
