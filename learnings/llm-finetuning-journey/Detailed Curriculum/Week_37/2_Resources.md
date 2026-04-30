# Week 37 Resources

## Papers

- [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560) — Wang et al. 2023. The foundational synthetic data generation method; your Tier 3 pipeline that generates SQL question/answer pairs from seed examples is a direct application of this technique.
- [Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing](https://arxiv.org/abs/2406.08464) — Xu et al. 2024. Shows that strong instruction data can be generated entirely from aligned models without seed tasks; relevant for understanding how to scale your 15K dataset beyond hand-written templates.
- [Self-Rewarding Language Models](https://arxiv.org/abs/2401.14367) — Yuan et al. 2024. Uses the model itself to generate reward signals for preference data; background for understanding how synthetic preference pairs can be built from your SQL generator without human labelers.

## Datasets

- [sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context) — 78K (question, schema, SQL) pairs. Best quality public dataset.
- [gretel/gretel-text-to-sql](https://huggingface.co/datasets/gretel/gretel-text-to-sql) — Synthetically generated, includes PostgreSQL-specific features.
- [Spider](https://yale-nlp.github.io/spider/) — 10K expert-labeled, cross-domain. Download from Yale NLP GitHub.
- [wikisql](https://huggingface.co/datasets/wikisql) — 80K simple single-table queries; good volume fallback.
- [Clinton/Text-to-sql-v1](https://huggingface.co/datasets/Clinton/Text-to-sql-v1) — Community aggregation dataset.
- [b-mc2/sql-create-context-v4-sql-prompt](https://huggingface.co/datasets/b-mc2/sql-create-context) — Extended version with more templates.

## Blog Posts / Articles

- [Synthetic Data Generation for SQL: Best Practices](https://defog.ai/blog/synthetic-data-for-sql/) — Defog AI. Practical guide for generating SQL training data.
- [How to Build a Text-to-SQL Model That Actually Works](https://medium.com/@lgeorge/how-to-build-a-text-to-sql-model) — Medium article on dataset quality for SQL fine-tuning.

## GitHub Repos

- [sql-eval by Defog AI](https://github.com/defog-ai/sql-eval) — An open-source framework for evaluating text-to-SQL models on PostgreSQL. Study this for Week 39's evaluation setup.
- [BIRD-SQL benchmark](https://bird-bench.github.io/) — A harder text-to-SQL benchmark using real-world databases. Useful as a stretch evaluation target.
- [Spider evaluation scripts](https://github.com/taoyds/spider) — Official evaluation code for Spider benchmark.

## Documentation

- [sqlparse documentation](https://sqlparse.readthedocs.io/) — Python SQL parsing library used for validation.
- [TimescaleDB documentation](https://docs.timescale.com/) — Reference for TimescaleDB-specific SQL functions; use when crafting hand-written examples.
- [HuggingFace datasets documentation](https://huggingface.co/docs/datasets/main/en/index) — How to load, filter, map, and push datasets.

## Videos

- [Sebastian Raschka — Finetuning Large Language Models](https://www.youtube.com/watch?v=iHrBQ4F6Gvs) — Sebastian Raschka — ~30 min. Covers data curation strategies, quality filtering, and format choices for SFT datasets; directly relevant to building your 15K SQL dataset.
- [Trelis Research — How to Create a Fine-tuning Dataset](https://www.youtube.com/watch?v=MtlHhNn2RDYQ) — Trelis Research — ~25 min. Practical walkthrough of synthetic data generation for instruction-following tasks; covers prompt templating and quality checks applicable to SQL.
- [Maxime Labonne — Data Curation for LLM Fine-Tuning](https://www.youtube.com/watch?v=XkMGrECUXQ4) — Maxime Labonne — ~30 min. Discusses deduplication, format validation, and dataset size trade-offs relevant to expanding to 15K examples.

## Optional / Bonus

- [DataComp: In Search of the Next Generation of Multimodal Datasets](https://arxiv.org/abs/2304.14108) — How dataset curation decisions affect model quality; principles transfer to text-to-SQL datasets.
- [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264) — How data repetition and quality affect training efficiency; relevant to your dataset size choice.
