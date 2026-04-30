# Week 57 Resources

## Papers

- [Don't Stop Pretraining: Adapt Language Models to Domains and Tasks](https://arxiv.org/abs/2004.10964) — Gururangan et al., 2020. The foundational paper on domain-adaptive pretraining (DAPT); establishes that CPT before fine-tuning consistently helps.
- [Efficient Online Data Mixing for Language Model Pre-Training](https://arxiv.org/abs/2312.02406) — Parmar et al., 2024. Dynamic data mixing during CPT; relevant if you want to balance your corpus sources.
- [CodeBERT: A Pre-Trained Model for Programming and Natural Languages](https://arxiv.org/abs/2002.08155) — Feng et al., 2020. Classic example of domain CPT for code; analogous to your SQL-domain CPT.
- [StarCoder: May the Source Be with You!](https://arxiv.org/abs/2305.06161) — Li et al., 2023. Large-scale code CPT; study their corpus construction methodology.

## Videos

- [Sebastian Raschka — Domain adaptation and continued pretraining](https://www.youtube.com/watch?v=iMMgivMDm8E) — Sebastian Raschka, ~35m.
- [Unsloth — Training on custom data](https://www.youtube.com/watch?v=2trUMnVVAhg) — Unsloth, ~30m. Walkthrough of Unsloth for CPT/SFT on RunPod.

## Blog Posts / Articles

- [Hugging Face — continued pretraining guide](https://huggingface.co/docs/transformers/en/model_doc/auto#continuing-pretraining) — Official documentation.
- [Tim Dettmers — GPU memory guide](https://timdettmers.com/2023/01/16/which-gpu-for-deep-learning/) — Understanding H100 vs A100 tradeoffs; relevant for RunPod selection.
- [RunPod — Getting started guide](https://docs.runpod.io/docs/getting-started) — Official RunPod setup documentation.

## GitHub Repos

- [unslothai/unsloth](https://github.com/unslothai/unsloth) — Fast training library; use their CPT examples notebook.
- [allenai/dolma](https://github.com/allenai/dolma) — The OLMo pretraining data pipeline; study their deduplication and quality filtering for CPT corpus building.
- [bigcode/the-stack-v2-train](https://huggingface.co/datasets/bigcode/the-stack-v2-train) — The Stack v2; filter for `.sql` extension.

## Documentation

- [PostgreSQL documentation download](https://www.postgresql.org/docs/) — Full PostgreSQL 16 docs available as HTML tarball.
- [TimescaleDB documentation source](https://github.com/timescale/docs) — TimescaleDB docs in Markdown format; easier to process than scraped HTML.
- [Stack Exchange data dump](https://archive.org/details/stackexchange) — The official Stack Overflow data dump; use the `Posts.xml` file for the PostgreSQL site.
- [Unsloth CPT documentation](https://github.com/unslothai/unsloth/wiki) — Unsloth's own guide for continued pretraining configurations.

## Optional / Bonus

- [LIMA follow-up: Does CPT help with alignment?](https://arxiv.org/abs/2309.01325) — Examines whether CPT and SFT are complementary for alignment.
- [Dolma: An Open Corpus of Three Trillion Tokens](https://arxiv.org/abs/2402.00159) — How large-scale corpus curation works at production scale.
- [SlimPajama: A 627B Token Clean and Deduplicated Version of RedPajama](https://huggingface.co/datasets/cerebras/SlimPajama-627B) — Study their quality filtering pipeline for CPT corpus inspiration.
