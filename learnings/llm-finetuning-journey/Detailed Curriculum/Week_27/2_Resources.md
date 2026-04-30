# Week 27 Resources — Phase 3 Gate

This week is self-assessment, not new material. The resources below serve two purposes: (1) reference links you should be able to navigate confidently as a Phase 3 graduate, and (2) remediation reading if any gate criterion flagged a gap.

---

## Papers

These are the papers you must be able to discuss at the level of a technical interview. No new reading required — use these for remediation only.

[Scaling Laws for Neural Language Models (Kaplan et al., 2020)](https://arxiv.org/abs/2001.08361) — The original OpenAI scaling law paper. If Chinchilla is still fuzzy, re-read Section 3.

[Training Compute-Optimal Large Language Models (Hoffmann et al., 2022)](https://arxiv.org/abs/2203.15556) — The Chinchilla paper. The key table is Table A3 (Approach 3 constants). You must be able to reproduce the N_opt / D_opt derivation.

[ZeRO: Memory Optimizations Toward Training Trillion Parameter Models (Rajbhandari et al., 2020)](https://arxiv.org/abs/1910.02054) — If ZeRO Stage 2 vs Stage 3 is still unclear, re-read Sections 2 and 3.

[The RefinedWeb Dataset for Falcon LLMs (Penedo et al., 2023)](https://arxiv.org/abs/2306.01116) — Data quality pipeline reference. If your dataset validity rate is below 98%, revisit the filtering approach described here.

[Self-Instruct: Aligning Language Models with Self-Generated Instructions (Wang et al., 2023)](https://arxiv.org/abs/2212.10560) — The Tier 3 synthetic generation technique used in your postgres-sql-v1 pipeline.

[Qwen2.5-Coder Technical Report (Hui et al., 2024)](https://arxiv.org/abs/2409.12186) — Your Phase 6 base model. You should be able to summarize the architecture and training setup without looking at your notes.

[DeepSeek-V3 Technical Report (DeepSeek, 2024)](https://arxiv.org/abs/2412.19437) — MoE architecture reference. Covers MLA (multi-head latent attention) and MTP (multi-token prediction) — terms you should now recognize.

---

## GitHub Repos

[lm-evaluation-harness (EleutherAI)](https://github.com/EleutherAI/lm-evaluation-harness) — The evaluation framework used in Week 23. You should have this installed and able to run `lm_eval --model hf --tasks hellaswag,arc_easy --device mps` on your 50M model.

[nanoGPT (Andrej Karpathy)](https://github.com/karpathy/nanoGPT) — The pretraining reference codebase. Your Week 20–21 code should be a recognizable descendant of this.

[HuggingFace Datasets](https://github.com/huggingface/datasets) — Used to host and load postgres-sql-v1. Criterion 3 requires a working Hub URL.

---

## Documentation

[HuggingFace Hub — Sharing Datasets](https://huggingface.co/docs/datasets/share) — How to push your dataset to the Hub with `push_to_hub()`. The dataset card (README.md with YAML front-matter) should be filled in as per Week 26 output.

[HuggingFace Accelerate](https://huggingface.co/docs/accelerate/index) — The distributed training wrapper used in Week 19. If you need to rerun training on multiple GPUs for remediation, this is the reference.

[sqlglot Dialect Docs](https://sqlglot.com/sqlglot/dialects/dialect.html) — The parser used for SQL validity checking in Criterion 3. PostgreSQL dialect is `"postgres"`.

---

## Videos

[Let's build GPT — from scratch, in code, spelled out (Andrej Karpathy, YouTube, 2:25:00)](https://www.youtube.com/watch?v=kCc8FmEb1nY) — The full pretraining walkthrough. If you struggled with Weeks 20–22, rewatch Sections 3–5 (the training loop and evaluation sections).

[Distributed Training with PyTorch (Meta AI, YouTube)](https://www.youtube.com/watch?v=14dmlCDMgLI) — Meta AI — ~35 min. Covers DDP and FSDP. Use for remediation if ZeRO / FSDP questions from Week 19 are still unclear.

---

## Blog Posts / Articles

[Sebastian Raschka — Understanding Large Language Model Training (magazine.sebastianraschka.com)](https://magazine.sebastianraschka.com/p/understanding-large-language-model) — Covers pretraining mechanics, data pipelines, and tokenization at a depth appropriate for Phase 3 gate review.

[HuggingFace Blog — How Large Language Models Are Trained](https://huggingface.co/blog/large-language-models) — End-to-end overview of the pretraining pipeline; good refresher on concepts covered in Weeks 17–21.

[Chinchilla Replication (Epoch AI)](https://epochai.org/blog/chinchilla-optimal-compute-is-a-good-fit-for-training-llms) — Independent verification of Chinchilla scaling predictions. Useful if you want to stress-test your understanding of Criterion 4.

---

## Optional / Bonus

[Phase 4 Preview: LoRA Paper (Hu et al., 2022)](https://arxiv.org/abs/2106.09685) — If you want to read ahead, this is the core LoRA paper you will study in Week 30. Read only after you have completed your gate evidence collection.

[Chinchilla Replication (Epoch AI blog post)](https://epochai.org/blog/chinchilla-optimal-compute-is-a-good-fit-for-training-llms) — An independent verification of Chinchilla scaling predictions. Useful if you want to stress-test your understanding of Criterion 4.
