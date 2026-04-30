# Week 61 Resources

## Papers

- [BIRD: A Big Bench for Large-Scale Database Grounded Text-to-SQL Evaluation](https://arxiv.org/abs/2305.03111) — Li et al., 2023. The BIRD-SQL benchmark paper; read for evaluation methodology.
- [Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task](https://arxiv.org/abs/1809.08887) — Yu et al., 2018. The original Spider paper.
- [Spider 2.0: Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows](https://arxiv.org/abs/2411.07763) — 2024. Enterprise SQL benchmark paper.
- [DAIL-SQL: Efficient Prompt Engineering for Text-to-SQL](https://arxiv.org/abs/2308.15363) — Gao et al., 2023. State-of-the-art prompting approach for BIRD and Spider; read for context on what you are competing against.

## Videos

- [Yale NLP — Spider benchmark introduction](https://www.youtube.com/watch?v=az3R9NRJFsQ) — Yale NLP, 15m.
- [AI Explained — Text to SQL deep dive](https://www.youtube.com/watch?v=U_pxVWPpNjQ) — AI Explained, 25m.

## Blog Posts / Articles

- [BIRD-SQL leaderboard](https://bird-bench.github.io/) — Current SOTA scores; see what you are competing against.
- [Defog SQLCoder model card and benchmark](https://defog.ai/blog/sqlcoder-7b/) — Defog's approach and benchmark numbers; your primary competitor.
- [Text-to-SQL evaluation best practices](https://medium.com/@kshitijhm/evaluation-metrics-for-text-to-sql-8b0ac60ebc4a) — Practical guide to evaluation metrics and their limitations.

## GitHub Repos

- [defog-ai/sql-eval](https://github.com/defog-ai/sql-eval) — Defog's open-source SQL evaluation framework.
- [taoyds/spider](https://github.com/taoyds/spider) — Official Spider dataset and evaluation scripts.
- [AlibabaResearch/DAIL-SQL](https://github.com/AlibabaResearch/DAIL-SQL) — DAIL-SQL implementation; study their eval harness for comparison.
- [bird-bench/mini-dev](https://github.com/bird-bench/mini-dev) — BIRD mini-dev set for quick iterations; 500 examples.

## Documentation

- [BIRD-SQL official download](https://bird-bench.github.io/) — Dataset download instructions.
- [Spider evaluation script](https://github.com/taoyds/spider/blob/master/evaluation.py) — Reference evaluation script; adapt for your harness.
- [HuggingFace evaluate library](https://huggingface.co/docs/evaluate/index) — Useful for metric computation and bootstrapping.

## Optional / Bonus

- [Spider 2.0 PostgreSQL subset](https://spider2-sql.github.io/) — The enterprise PostgreSQL benchmark; most relevant to your domain.
- [WikiSQL: A Large Crowd-Sourced Dataset for Developing Natural Language Interfaces](https://arxiv.org/abs/1709.00103) — The simpler predecessor to Spider; run your model on it as a sanity check (your model should score > 90%).
