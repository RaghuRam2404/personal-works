# Week 44 Resources

## Papers

- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — Bai et al. (Anthropic) 2022. Read abstract and Section 2 ("Constitutional AI Principles"). The main takeaway for this week: structured principles can replace human judgment for preference labeling.
- [RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback](https://arxiv.org/abs/2309.00267) — Lee et al. 2023. Demonstrates that AI feedback at scale can match or exceed human feedback quality. Skim the methodology section.

## Blog Posts / Articles

- [How to Build a Preference Dataset for DPO](https://huggingface.co/blog/pref-tuning) — HuggingFace. Covers data collection strategies including synthetic generation and execution-based labeling.
- [Generating Synthetic Data for DPO Fine-Tuning](https://huggingface.co/blog/synthetic-data-save-costs) — HuggingFace. Discusses cost-effective approaches to building preference datasets without expensive human annotation.

## GitHub Repos / Datasets

- [Spider Benchmark](https://github.com/taoyds/spider) — The standard Text-to-SQL benchmark. Use Spider prompts adapted to your schema as your starting prompt set.
- [WikiSQL Dataset](https://github.com/salesforce/WikiSQL) — Simpler single-table SQL benchmark. Good for generating easy/medium preference pairs.
- [HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) — Reference example of a well-structured DPO preference dataset. Study its schema and dataset card.
- [b-mc2/sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context) — SQL generation dataset with schema context. Good source for prompts and reference SQL.

## Documentation

- [psycopg2 documentation](https://www.psycopg.org/docs/) — The Python Postgres adapter. Focus on connection handling, cursor methods, and error types.
- [PostgreSQL statement_timeout](https://www.postgresql.org/docs/current/runtime-config-client.html#GUC-STATEMENT-TIMEOUT) — Postgres docs for the statement timeout setting.
- [HuggingFace datasets.push_to_hub](https://huggingface.co/docs/datasets/upload_dataset) — How to push a Dataset to HuggingFace Hub programmatically.

## Videos

- [Yannic Kilcher — Constitutional AI: Harmlessness from AI Feedback (Anthropic)](https://www.youtube.com/watch?v=jit9_QZLE6U) — Yannic Kilcher — ~35 min. Paper walkthrough of the Constitutional AI paper; explains the critique-revision loop and how AI-generated principles replace human annotators for preference labeling.
- [AI Coffee Break — RLAIF: Reinforcement Learning from AI Feedback](https://www.youtube.com/watch?v=c-Kx_EMX-0k) — AI Coffee Break with Letitia — ~20 min. Accessible explainer of how AI feedback scales preference labeling; directly applicable to your SQL preference dataset generation pipeline.
- [HuggingFace — Preference Datasets and DPO](https://www.youtube.com/watch?v=TjVMEJqHOVA) — HuggingFace — ~30 min. Covers the structure of a well-formed preference dataset (chosen/rejected format) and how execution-based labeling fits into the pipeline.

## Optional / Bonus

- [Self-Play Fine-Tuning (SPIN)](https://arxiv.org/abs/2401.01335) — An approach to generating preference data from a single model by having it compete against itself. Alternative to cross-model comparison.
- [Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs](https://arxiv.org/abs/2406.08464) — Technique for generating high-quality instruction-following data from aligned models. Useful for Phase 6 dataset scaling.
- [Executable Code Actions Elicit Better LLM Agents](https://arxiv.org/abs/2402.01030) — Shows why executable verification is a stronger signal than text-based evaluation for code/SQL tasks.
