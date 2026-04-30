# Week 53 Resources

## Papers

- [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) — Zhou et al., 2023. The empirical case for quality over quantity in alignment fine-tuning.
- [Tulu 3: Pushing Frontiers in Open Language Model Post-Training](https://arxiv.org/abs/2411.15124) — Lambert et al., 2024. AllenAI's fully documented data curation + training pipeline.
- [DEITA: What Makes Good Data for Alignment?](https://arxiv.org/abs/2312.15685) — Liu et al., 2023. Complexity and quality scoring for instruction data selection.
- [Alpagasus: Training a Better Alpaca with Fewer Data](https://arxiv.org/abs/2307.08701) — Chen et al., 2023. LLM-as-judge for filtering instruction data; precursor to your Week 55 work.
- [Data Selection for Language Models via Importance Resampling](https://arxiv.org/abs/2302.03169) — Xie et al., 2023. DSIR: a principled approach to data selection using n-gram statistics.

## Videos

- [Andrej Karpathy — State of GPT (Microsoft Build 2023)](https://www.youtube.com/watch?v=bZQun8Y4L2A) — Microsoft, 42m. Covers SFT → RLHF pipeline; Section on data quality is directly relevant.
- [Yannic Kilcher — LIMA paper walkthrough](https://www.youtube.com/watch?v=vEFRHqf_Tts) — Yannic Kilcher, ~30m.

## Blog Posts / Articles

- [Tulu 3 blog post (AllenAI)](https://allenai.org/blog/tulu-3-technical-details) — Extended discussion of data decisions not fully covered in the paper.
- [Data-Centric AI Resource Hub](https://datacentricai.org/) — Collection of resources on data quality, curation, and labeling.
- [Lilian Weng — Prompt Engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/) — Relevant to teacher prompt design for Week 54.

## GitHub Repos

- [allenai/open-instruct](https://github.com/allenai/open-instruct) — The code behind Tulu 3; study their data preprocessing pipeline.
- [datasketch](https://github.com/ekzhu/datasketch) — MinHash and LSH implementation used for deduplication.
- [sqlglot](https://github.com/tobymao/sqlglot) — SQL parser and transpiler; use for AST depth and dialect conversion.
- [HuggingFace datasets deduplication](https://github.com/huggingface/datatrove) — DataTrove pipeline used by HF for large-scale deduplication.

## Documentation

- [HuggingFace Datasets documentation](https://huggingface.co/docs/datasets/index) — For loading, filtering, and pushing your v3 dataset.
- [PostgreSQL SQL syntax reference](https://www.postgresql.org/docs/current/sql.html) — Ground truth for what valid PostgreSQL looks like.
- [TimescaleDB documentation — hyperfunctions](https://docs.timescale.com/use-timescale/latest/hyperfunctions/) — Reference for TimescaleDB-specific SQL constructs.

## Optional / Bonus

- [The Data-Centric AI movement (Andrew Ng's framing)](https://datacentricai.org/blog/the-data-centric-ai-competition/) — Broader context for why data quality is the competitive moat.
- [Dolma dataset paper](https://arxiv.org/abs/2402.00159) — OLMo's pretraining data; useful to understand what large-scale curation looks like.
- [RedPajama-Data-v2](https://github.com/togethercomputer/RedPajama-Data) — Open pretraining corpus with quality signal annotations; study their filtering pipeline.
