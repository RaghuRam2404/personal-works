# Week 29 Resources

## Papers

- [InstructGPT: Training Language Models to Follow Instructions](https://arxiv.org/abs/2203.02155) — Ouyang et al. 2022. SFT is stage 1 of this pipeline.

## Videos

- [Fine-tuning Large Language Models (SFT Tutorial)](https://www.youtube.com/watch?v=iHrBQ4F6Gvs) — Sebastian Raschka — ~30 min. Practical walkthrough of SFT with SFTTrainer, covering ChatML formatting, dataset preparation, and key hyperparameters for Qwen-family models.
- [How to Fine-Tune an LLM with TRL's SFTTrainer](https://www.youtube.com/watch?v=RM3BO3cMFrc) — HuggingFace — ~25 min. Official HuggingFace tutorial demonstrating SFTTrainer configuration, packing, and logging to W&B; directly mirrors this week's assignment.
- [First Steps with SFT: Training a Small Language Model](https://www.youtube.com/watch?v=1Hgo2FmKwO8) — Trelis Research — ~20 min. End-to-end demonstration of full SFT on a sub-1B model using a small dataset subset; covers common gotchas with chat templates and EOS tokens.

## Blog Posts / Articles

- [HuggingFace LLM Course Chapter 11: Supervised Fine-Tuning](https://huggingface.co/learn/llm-course/chapter11/1) — Walkthrough of SFT concepts. Required reading.
- [HuggingFace Fine-Tuning Tutorial](https://huggingface.co/docs/transformers/training) — Transformers library official fine-tuning guide.
- [Qwen2.5 Model Card and Chat Template Details](https://huggingface.co/Qwen/Qwen2.5-0.5B) — Inspect the tokenizer config to understand the ChatML format.

## GitHub Repos

- [TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl) — HuggingFace. Contains `SFTTrainer`, `SFTConfig`, and example notebooks.
- [TRL SFT examples](https://github.com/huggingface/trl/tree/main/examples/scripts) — Look for `sft.py` for a minimal reference implementation.

## Documentation

- [TRL SFTTrainer docs](https://huggingface.co/docs/trl/sft_trainer) — Complete parameter reference. Know `dataset_text_field`, `packing`, `max_seq_length`, `neftune_noise_alpha`.
- [HuggingFace Trainer docs](https://huggingface.co/docs/transformers/main_classes/trainer) — `SFTTrainer` inherits all `TrainingArguments`; read the LR scheduler and optimizer options.

## Datasets

- [sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context) — 78K (question, context, answer) examples. Use 1K for this week.
- [Spider dataset](https://yale-nlp.github.io/spider/) — Cross-domain text-to-SQL benchmark; good for diverse schemas.
- [wikisql](https://huggingface.co/datasets/wikisql) — Simpler SQL dataset; good for quick prototyping but limited to single-table queries.

## Optional / Bonus

- [NEFTune: Noisy Embeddings Improve Instruction Finetuning](https://arxiv.org/abs/2310.05914) — A simple trick (add noise to embeddings during SFT) that often improves chat quality. Available in `SFTConfig` via `neftune_noise_alpha`.
- [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264) — What happens when you train on repeated data; relevant for understanding epoch count choices in SFT.
