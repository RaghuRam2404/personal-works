# Week 51 Resources

## Papers

- [Overfitting to Evaluation Sets in NLP](https://arxiv.org/abs/2005.00211) — Roelofs et al. 2020. How evaluation sets get "overfit" through repeated use; directly relevant to the Week 51 blind validation requirement.
- [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374) — Chen et al. 2021 (HumanEval paper). The pass@k eval methodology is analogous to your execution accuracy metric; the eval contamination discussion is directly relevant.

## Videos

- [Hamel Husain — Your AI Product Needs Evals](https://www.youtube.com/watch?v=r4kIsj6fyeo) — Hamel Husain — ~45 min. Practical framework for building evaluation pipelines that catch real model failures; directly applicable to your iteration decision-making this week.
- [Eugene Yan — Applied ML: From Experimentation to Production](https://www.youtube.com/watch?v=5c2bRSVzDmo) — Eugene Yan — ~40 min. Covers the model selection and iteration discipline needed to pick the best checkpoint from multiple runs.

## Blog Posts / Articles

- [Model Evaluation Best Practices](https://huggingface.co/docs/evaluate/index) — HuggingFace Evaluate library documentation. For building a rigorous eval pipeline that can be run consistently across all model versions.
- [How to Write a Good Model Card](https://huggingface.co/docs/hub/model-cards) — HuggingFace. Guide to documenting your model for HuggingFace Hub. Your `postgres-sqlcoder-7b-phase5-best` needs a model card.

## GitHub Repos

- [huggingface/trl](https://github.com/huggingface/trl) — TRL training library source. The `examples/` directory has eval scripts you can adapt to run consistent evaluation across your v3/v3-iter1/v3-iter2 checkpoints.
- [unslothai/unsloth](https://github.com/unslothai/unsloth) — Reference for any memory-efficient inference needed when running your full eval suite against 7B checkpoints on a limited VRAM budget.
- [tobymao/sqlglot](https://github.com/tobymao/sqlglot) — SQL parser used for building more sophisticated eval metrics (GROUP BY presence check, table name verification against schema).

## Tools

- [HuggingFace Hub model versioning](https://huggingface.co/docs/hub/repositories-tags) — How to tag model versions on the Hub (e.g., tagging `phase5-best` as v0.5.0). Useful for Phase 6 version management.
- [W&B Compare Runs](https://docs.wandb.ai/guides/runs/compare) — How to create side-by-side run comparisons in W&B for your iteration experiments.

## Optional / Bonus

- [The ML Test Score: A Rubric for Production Readiness](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/aab7d5b51ae7478a9e4a6f44a7b78e4ddde0a13e.pdf) — Google. A checklist for production ML systems. As you approach Phase 6 deployment, this rubric becomes relevant.
- [Spider 2.0 Leaderboard](https://spider2-sql.github.io/) — The state-of-the-art SQL benchmark. Compare your Phase 5 best model's accuracy against the leaderboard to calibrate how close you are to SOTA.
