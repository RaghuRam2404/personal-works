# Week 38 Resources

## Papers

- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Dettmers et al. 2023. The method you are applying this week; re-read Section 2 (NF4 quantization and double quantization) and Section 4 (results showing QLoRA matches full SFT quality) before starting your training run.
- [Qwen2.5-Coder Technical Report](https://arxiv.org/abs/2409.12186) — Hui et al. 2024. Your base model; re-read the architecture and training data sections to understand what capabilities the base already has before your domain fine-tune.
- [LoRA Land: Fine-Tuned Open-Source LLMs that Outperform GPT-4](https://arxiv.org/abs/2405.00732) — Guo et al. 2024. Systematic study of LoRA fine-tuned models across 25 tasks; provides reference numbers for what execution accuracy is achievable with QLoRA on code-generation tasks.

## Blog Posts / Articles

- [How to Fine-Tune LLMs with Unsloth: Step-by-Step](https://unsloth.ai/blog/finetune) — Unsloth blog. Official walkthrough; reference during your training run.
- [Understanding Your Training Metrics](https://wandb.ai/site/articles/understanding-your-training-metrics) — W&B blog. Guide to interpreting training loss curves, gradient norms, and eval metrics.

## GitHub Repos

- [Unsloth notebooks](https://github.com/unslothai/unsloth/tree/main/notebooks) — Find the Qwen2.5 notebook; adapt it to your dataset.
- [sql-eval by Defog AI](https://github.com/defog-ai/sql-eval) — Evaluation framework you will use in Week 39. Download and study this week so you understand what you are building toward.

## Documentation

- [HuggingFace model card guide](https://huggingface.co/docs/hub/model-cards) — How to write a proper model card; includes templates.
- [HuggingFace model tags](https://huggingface.co/docs/hub/model-tags) — How to tag your model for discoverability (`text2sql`, `postgresql`, etc.)
- [PEFT PeftModel.push_to_hub](https://huggingface.co/docs/peft/package_reference/peft_model#peft.PeftModel.push_to_hub) — How to push adapter-only to Hub.

## Evaluation Tools

- [sqlparse](https://sqlparse.readthedocs.io/) — For quick SQL validation during evaluation.
- [psycopg2](https://www.psycopg.org/docs/) — PostgreSQL Python adapter; used in Week 39's execution-based eval.
- [BIRD-SQL benchmark](https://bird-bench.github.io/) — A harder benchmark; run your model here after Week 39 as a stretch goal.

## Videos

- [Daniel Han — Unsloth QLoRA Fine-Tuning Tutorial](https://www.youtube.com/watch?v=aQmoog_s8_k) — Unsloth AI — ~20 min. Step-by-step walkthrough of Unsloth + TRL SFTTrainer on a 7B model; covers the exact setup you are using this week on the A100.
- [AI Anytime — Fine-Tune Qwen2.5 with Unsloth](https://www.youtube.com/watch?v=oHQ4J7bvGCM) — AI Anytime — ~25 min. Practical tutorial covering Unsloth notebook setup, VRAM usage, and pushing the adapter to HuggingFace Hub after training.
- [Sebastian Raschka — Practical LLM Fine-Tuning](https://www.youtube.com/watch?v=iHrBQ4F6Gvs) — Sebastian Raschka — ~30 min. Covers LoRA hyperparameter choices and how to read training loss curves during a QLoRA run; useful calibration for your A100 training session.

## Optional / Bonus

- [Text-to-SQL Benchmarking](https://arxiv.org/abs/2408.05109) — Survey of evaluation methods for text-to-SQL models; useful context for understanding the limitations of exact match.
- [Defog's SQLCoder model card](https://huggingface.co/defog/sqlcoder-7b-2) — A real production text-to-SQL model on HuggingFace. Compare their model card structure to what you are building.
