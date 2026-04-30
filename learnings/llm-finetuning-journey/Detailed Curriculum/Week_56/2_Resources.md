# Week 56 Resources

## Papers

- [CoSQL: A Conversational Text-to-SQL Challenge](https://arxiv.org/abs/1909.05378) — Yu et al., 2019. The primary conversational SQL dataset.
- [SParC: Cross-Domain Semantic Parsing in Context](https://arxiv.org/abs/1906.02285) — Yu et al., 2019. Sequential question refinement dataset.
- [ATIS: Airline Travel Information System](https://aclanthology.org/H90-1021.pdf) — Classic multi-turn NL-to-SQL dataset; historical context.
- [CHASE: A Large-Scale and Pragmatic Chinese Dataset for Cross-Database Context-Dependent Text-to-SQL](https://arxiv.org/abs/2108.00472) — Larger multi-turn dataset; useful for understanding the format.

## Videos

- [Stanford NLP Group — Context-dependent Semantic Parsing](https://www.youtube.com/watch?v=6p2pN0-vwCI) — Stanford NLP — ~25 min. Overview of the SParC/CoSQL research program and the challenges of cross-turn context resolution.

## Blog Posts / Articles

- [Hugging Face — Training on conversational data with TRL](https://huggingface.co/docs/trl/sft_trainer#training-on-completions-only) — Official docs for DataCollatorForCompletionOnlyLM.
- [Lilian Weng — Neural machine translation and sequence-to-sequence](https://lilianweng.github.io/posts/2018-06-24-attention/) — Background on multi-turn sequence modeling.
- [Defog blog — multi-turn SQL models](https://defog.ai/blog/multi-turn-sql/) — Practical experience with conversational SQL fine-tuning.

## GitHub Repos

- [taoyds/cosql](https://github.com/taoyds/cosql) — Official CoSQL repository with train/dev splits.
- [taoyds/sparc](https://github.com/taoyds/sparc) — Official SParC repository.
- [tobymao/sqlglot](https://github.com/tobymao/sqlglot) — SQL dialect transpilation; essential for SQLite → PostgreSQL conversion.
- [dimitri/pgloader](https://github.com/dimitri/pgloader) — Migrating SQLite databases to PostgreSQL for CoSQL validation.
- [huggingface/trl](https://github.com/huggingface/trl) — DataCollatorForCompletionOnlyLM and SFTTrainer for multi-turn training.

## Documentation

- [TRL SFTTrainer — dataset_text_field and dataset_kwargs](https://huggingface.co/docs/trl/sft_trainer) — How to pass chat-format datasets to SFTTrainer.
- [Qwen2.5 chat template](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct#chatml-format) — The exact chat template format for your base model.
- [TimescaleDB time_bucket_gapfill](https://docs.timescale.com/api/latest/hyperfunctions/gapfilling/time_bucket_gapfill/) — Reference for the most common multi-turn refinement in TimescaleDB.

## Optional / Bonus

- [MISP: Understanding the natural language understanding of code intelligence models](https://arxiv.org/abs/2305.02049) — Work on implicit reference resolution in code tasks; similar challenges to implicit SQL context.
- [IGSQL: Database Schema Interaction Graph Based Neural Model](https://arxiv.org/abs/2011.05744) — Early work on schema linking across turns; useful background reading.
