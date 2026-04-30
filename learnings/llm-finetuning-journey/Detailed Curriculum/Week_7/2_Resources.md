# Week 7 — Resources

## Papers

- [HuggingFace Transformers: State-of-the-Art Natural Language Processing](https://arxiv.org/abs/1910.03771) — Wolf et al. 2020. The paper introducing the transformers library; explains the unified model hub, pipeline API, and AutoClass design. Read the abstract and Section 2.
- [Datasets: A Community Library for Natural Language Processing](https://arxiv.org/abs/2109.02846) — Lhoest et al. 2021. The paper behind the `datasets` library you are using this week; covers memory-mapped Arrow storage and the hub integration.

## Videos

- [HuggingFace NLP Course — What is Transfer Learning?](https://www.youtube.com/watch?v=BqqLezdIUsI) — HuggingFace — ~9 min. Concise intro to transfer learning and why HuggingFace pretrained models work out of the box; watch before starting Chapter 1.
- [HuggingFace NLP Course — The pipeline function](https://www.youtube.com/watch?v=tiZFewofSLM) — HuggingFace — ~6 min. Walks through the `pipeline()` API you will call in this week's assignment.
- [Lewis Tunstall — Fine-tuning LLMs with the HuggingFace Ecosystem](https://www.youtube.com/watch?v=us_v-HPSMqQ) — Lewis Tunstall (HuggingFace) — ~45 min. End-to-end walkthrough of the Trainer API, tokenizers, and Datasets; directly mirrors what you implement this week.
- [Sasha Rush — MiniTorch and the HuggingFace Pipeline](https://www.youtube.com/watch?v=vAGPkLSPRaU) — Sasha Rush — ~30 min. Covers how the HuggingFace model hub is structured and how AutoModel dispatch works under the hood.

## Courses / Tutorials

- [HuggingFace LLM Course — Chapters 1–3](https://huggingface.co/learn/llm-course/chapter1/1) — **Required. Do every exercise.** Chapter 1: pipeline and AutoClass. Chapter 2: model internals and tokenizers. Chapter 3: fine-tuning with `Trainer`. This is the canonical HuggingFace onboarding resource.
- [HuggingFace `datasets` Tutorial](https://huggingface.co/docs/datasets/tutorial) — Covers loading, map, filter, push_to_hub, and save_to_disk. Read Sections 1–4.

## Documentation

- [transformers AutoModelForCausalLM docs](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForCausalLM) — Full parameter list for `from_pretrained`.
- [transformers generation docs](https://huggingface.co/docs/transformers/generation_strategies) — Covers greedy, beam search, top-k, top-p, temperature. Read "Generation strategies" section in full.
- [datasets library docs](https://huggingface.co/docs/datasets/index) — Reference for `.map()`, `.filter()`, `.select()`, `.save_to_disk()`, `.push_to_hub()`.
- [huggingface_hub library docs](https://huggingface.co/docs/huggingface_hub/index) — For `login()`, `push_to_hub()`, and programmatic Hub access.

## Blog Posts / Articles

- [The HuggingFace Ecosystem Explained](https://huggingface.co/blog/transformers-pytorch-2) — Overview of transformers + PyTorch 2.0 integration, including `torch.compile`.
- [Fine-tuning a Pretrained Model (HuggingFace Blog)](https://huggingface.co/docs/transformers/training) — Step-by-step fine-tuning with `Trainer`. Read this even though you won't use `Trainer` until Phase 4.
- [How to Generate Text: Different Decoding Methods](https://huggingface.co/blog/how-to-generate) — Excellent deep dive into greedy, beam, top-k, and nucleus sampling with examples.

## Models Referenced This Week

- [distilgpt2](https://huggingface.co/distilgpt2) — 82M param GPT-2 distilled model. Fast, no gating, good for exploring the API.
- [Qwen/Qwen2.5-Coder-7B](https://huggingface.co/Qwen/Qwen2.5-Coder-7B) — The target model for your fine-tuning capstone. Download the tokenizer now; the full model is 14GB and requires Colab Pro or better.
- [Qwen/Qwen2.5-Coder-1.5B](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B) — Smaller version, fits on Colab Free. Good for Week 7 experiments.

## Datasets Referenced This Week

- [Spider on HuggingFace Hub](https://huggingface.co/datasets/spider) — Text-to-SQL dataset. `load_dataset("spider")`.
- [BIRD-SQL](https://bird-bench.github.io/) — More complex text-to-SQL benchmark. Start downloading now; you will use it in Phase 5.

## GitHub Repos

- [huggingface/transformers](https://github.com/huggingface/transformers) — Source code for the transformers library. When something breaks, reading the source is the fastest debugging path.
- [huggingface/datasets](https://github.com/huggingface/datasets) — Source for the datasets library. The `examples/` folder has usage patterns for every major dataset format.

## Optional / Bonus

- [The Illustrated BERT, ELM, and co.](https://jalammar.github.io/illustrated-bert/) — Jay Alammar's visual guide to BERT. Not directly required this week but gives intuition for encoder-only models (contrast to GPT-style decoders).
- [HuggingFace Spaces](https://huggingface.co/spaces) — Interactive demo hosting. Not required now, but in Phase 6 you will deploy your fine-tuned SQL model here.
