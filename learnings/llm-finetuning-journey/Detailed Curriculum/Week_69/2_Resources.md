# Week 69 Resources — Evaluation and Ablations

## Papers

[BIRD: A Big Bench for Large-Scale Database Grounded Text-to-SQL Evaluation](https://arxiv.org/abs/2305.03111) — Li et al. 2023; understand the benchmark you compare on; execution accuracy definition is in Section 3.

[Spider: A Large-Scale Human-Labeled Dataset](https://arxiv.org/abs/1809.08887) — Yu et al. 2018; exact-match evaluation protocol defined here.

[Defog SQL Eval](https://github.com/defog-ai/sql-eval) — Not a paper but the evaluation codebase you use; the README explains the scoring methodology.

[Beyond the Imitation Game: Quantifying and Extrapolating the Capabilities of Language Models (BIG-Bench)](https://arxiv.org/abs/2206.04615) — Srivastava et al. 2022; good reference for how to present multi-benchmark evaluation tables at scale.

## Videos

[How to Design a Machine Learning Experiment (Andrej Karpathy, informal)](https://www.youtube.com/watch?v=PaCmpygFfXo) — ~1h; touches on ablations, baselines, and what makes an evaluation trustworthy.

## Blog Posts / Articles

[Evaluation Hell — Understanding and Avoiding Pitfalls in LLM Evals (Lilian Weng)](https://lilianweng.github.io/posts/2024-02-05-human-data-quality/) — Covers contamination, metric selection, and evaluation reproducibility for LLMs.

[Statistical Significance Testing in ML Papers (Elvis Dohmatob)](https://arxiv.org/abs/2010.09875) — When to use hypothesis testing, how to compute CIs for accuracy metrics, and why most ML papers skip this (and when that is acceptable).

## GitHub Repos

[defog-ai/sql-eval](https://github.com/defog-ai/sql-eval) — The Defog evaluation framework; includes execution accuracy against a PostgreSQL backend.

[spider/evaluation.py](https://github.com/taoyds/spider/blob/master/evaluation.py) — Spider's official exact-match evaluation script; use this for your Spider 1.0 numbers.

[DAIL-SQL evaluation](https://github.com/BeachWang/DAIL-SQL) — A recent high-performing text-to-SQL system with a clean evaluation pipeline you can adapt.

## Documentation

[SciPy stats.binom_test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html) — For computing exact binomial confidence intervals on your accuracy estimates: `binomtest(k=166, n=200, p=0.83).proportion_ci()`.

## Optional / Bonus

[HELM: Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110) — Liang et al. 2022; large-scale evaluation framework; valuable for understanding how to structure multi-benchmark comparisons.

[Chatbot Arena: Judging LLMs as a Judge](https://arxiv.org/abs/2403.04132) — Zheng et al. 2024; explains pairwise evaluation as an alternative to absolute benchmarks; useful if you want to add a human preference evaluation to your report.
