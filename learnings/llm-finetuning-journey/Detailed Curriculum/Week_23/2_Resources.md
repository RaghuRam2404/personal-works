# Week 23 Resources — LM Evaluation

## Papers

- [EleutherAI lm-evaluation-harness: A Framework for Few-Shot Evaluation](https://arxiv.org/abs/2404.12253) — Gao et al. 2024. The paper behind the tool you are using this week.
- [Measuring Massive Multitask Language Understanding (MMLU)](https://arxiv.org/abs/2009.03300) — Hendrycks et al. 2021. Read the benchmark methodology and subject breakdown.
- [HellaSwag: Can a Machine Really Finish Your Sentence?](https://arxiv.org/abs/1905.07830) — Zellers et al. 2019. The paper introducing HellaSwag and explaining the adversarial filtering approach.
- [Think you have Solved Question Answering? Try ARC](https://arxiv.org/abs/1803.05457) — Clark et al. 2018. Introduction to ARC benchmarks.
- [Beyond the Imitation Game: Quantifying and Extrapolating the Capabilities of Language Models (BIG-Bench)](https://arxiv.org/abs/2206.04615) — Srivastava et al. 2022. A larger benchmark collection; useful for understanding what tasks are measured.

## Videos

- [Andrej Karpathy — State of GPT (Microsoft Build 2023)](https://www.youtube.com/watch?v=bZQun8Y4L2A) — (~1h). The evaluation section (minutes 20–40) explains benchmark methodology and what the numbers mean.
- [HELM: Holistic Evaluation of Language Models](https://www.youtube.com/watch?v=GThW4oQOFoQ) — Stanford CRFM presentation (~30 min).

## Blog Posts / Articles

- [HELM: Holistic Evaluation of Language Models](https://crfm.stanford.edu/helm/latest/) — Stanford CRFM. Read the methodology section on multi-metric evaluation.
- [How to avoid contamination when benchmarking LLMs](https://huyenchip.com/2024/03/28/llm-benchmark-contamination.html) — Chip Huyen. Practical guide to detecting and avoiding benchmark contamination.
- [What's in MMLU?](https://blog.eleuther.ai/mulitask-lm-evaluation-harness-update/) — EleutherAI. Discussion of MMLU's strengths and limitations.

## GitHub Repos

- [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — Required for this week's assignment. Read the README before installing.
- [stanford-crfm/helm](https://github.com/stanford-crfm/helm) — HELM evaluation framework; an alternative to lm-eval with different benchmark coverage.
- [yale-lily/spider](https://github.com/taoyds/spider) — Spider text-to-SQL benchmark; relevant for your Phase 6 evaluation.

## Documentation

- [lm-eval task list](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_table.md) — Full list of available tasks. Browse before choosing what to run.
- [lm-eval custom model integration](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/model_guide.md) — Required if your model is not a standard HuggingFace model.

## Optional / Bonus

- [BIRD: A Big Bench for Large-scale Database Grounded Text-to-SQLs](https://arxiv.org/abs/2305.03111) — Li et al. 2023. A harder text-to-SQL benchmark than Spider; relevant for Phase 6.
- [Spider 2.0](https://spider2-sql.github.io/) — A harder, more realistic text-to-SQL benchmark; uses real enterprise databases including PostgreSQL.
- [Evaluating Large Language Models: A Survey](https://arxiv.org/abs/2307.03109) — Chang et al. 2023. Comprehensive survey of LLM evaluation methodologies.
