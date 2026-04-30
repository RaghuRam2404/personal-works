# Week 25 Resources — Dataset Construction

## Papers

- [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560) — Wang et al. 2022. The foundational paper for synthetic instruction generation. Read all sections.
- [Alpaca: A Strong, Replicable Instruction-Following Model](https://crfm.stanford.edu/2023/03/13/alpaca.html) — Taori et al. 2023, Stanford CRFM. Blog post + paper on using Self-Instruct to create fine-tuning data.
- [BIRD: A Big Bench for Large-scale Database Grounded Text-to-SQLs](https://arxiv.org/abs/2305.03111) — Li et al. 2023. Harder text-to-SQL benchmark with real databases; your Tier 1 source.
- [Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing](https://arxiv.org/abs/1809.08887) — Yu et al. 2018. The Spider benchmark paper; read Section 2 for dataset construction methodology.

## Videos

- [How to create fine-tuning datasets](https://www.youtube.com/watch?v=XkMGrECUXQ4) — Maxime Labonne, ~30 min. Practical guide to creating SFT datasets.
- [Building good LLM training datasets](https://www.youtube.com/watch?v=yBL7J0kgldU) — Yannic Kilcher — ~10 min. Short but relevant overview of dataset quality considerations.

## Blog Posts / Articles

- [Alpaca GitHub README](https://github.com/tatsu-lab/stanford_alpaca) — Stanford. The Alpaca format definition and Self-Instruct pipeline.
- [ShareGPT Vicuna dataset card](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) — HuggingFace. Shows the ShareGPT format in practice.
- [Maxime Labonne's LLM dataset guide](https://mlabonne.github.io/blog/posts/2024-02-26-Fine_tune_a_small_language_model.html) — Comprehensive guide to dataset formats for fine-tuning.
- [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) — Zhou et al. 2023. Why 1,000 high-quality examples can outperform 50K low-quality ones.

## GitHub Repos

- [tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca) — Alpaca dataset and Self-Instruct pipeline. Reference implementation.
- [lm-sys/FastChat](https://github.com/lm-sys/FastChat) — Vicuna and ShareGPT format. The data processing scripts are useful.
- [taoyds/spider](https://github.com/taoyds/spider) — Spider dataset; includes schema information and train/dev/test splits.
- [bird-bench/mini-dev](https://github.com/bird-bench/mini-dev) — BIRD mini-dev set; smaller subset for rapid iteration.
- [tobymao/sqlglot](https://github.com/tobymao/sqlglot) — SQL parsing and transpilation. Required for your quality filter.

## Documentation

- [HuggingFace datasets documentation](https://huggingface.co/docs/datasets/) — How to load Spider, BIRD, and custom datasets.
- [TRL SFTTrainer documentation](https://huggingface.co/docs/trl/sft_trainer) — How to use `DataCollatorForCompletionOnlyLM` for loss masking.
- [Qwen2.5-Coder chat template](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct/blob/main/tokenizer_config.json) — The exact ChatML template your model expects; your data must match this format.

## Optional / Bonus

- [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) — Meta AI 2023. Demonstrates that 1,000 high-quality curated examples can outperform much larger datasets.
- [Orca: Progressive Learning from Complex Explanation Traces](https://arxiv.org/abs/2306.02707) — Microsoft 2023. How synthetic explanation-augmented data improves reasoning.
- [WizardLM: Empowering Large Language Models to Follow Complex Instructions](https://arxiv.org/abs/2304.12244) — Xu et al. 2023. Evol-Instruct method for making synthetic instructions progressively harder.
