# Week 17 Resources — Scaling Laws

## Papers

- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — Kaplan et al. 2020. The original scaling law paper from OpenAI. Read Sections 1–4.
- [Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556) — Hoffmann et al. 2022. The paper that revised Kaplan. Focus on Section 3 and Table A3.
- [Cramming: Training a Language Model on a Single GPU in One Day](https://arxiv.org/abs/2212.14034) — Geiping & Goldstein 2022. Shows how far you can push a small compute budget; relevant for your Week 20–22 project.
- [Scaling Laws for Fine-Tuning](https://arxiv.org/abs/2206.07660) — Zhao et al. 2022. How fine-tuning scales differently from pretraining.

## Videos

- [Chinchilla paper walkthrough](https://www.youtube.com/watch?v=PZXN7jTLnso) — Yannic Kilcher (~30 min). Required watching. Clear explanation of the IsoFLOP methodology.
- [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU) — Andrej Karpathy (~4h). Skim for the sections on compute budgeting.

## Blog Posts / Articles

- [A Hitchhiker's Guide to Scaling Laws](https://www.lesswrong.com/posts/midXmMb2Xg37F2Kgn/a-hitchhiker-s-guide-to-scaling-laws) — LessWrong post with clear derivations and worked examples.
- [Chinchilla's Wild Implications](https://www.lesswrong.com/posts/6Fpvch8RR29qLEWNH/chinchilla-s-wild-implications) — LessWrong. Thoughtful analysis of what Chinchilla means for the field.
- [EleutherAI Scaling Laws post](https://blog.eleuther.ai/scaling-laws/) — EleutherAI's take on what the scaling law research actually implies.

## GitHub Repos

- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) — You will use this as a starting point for your Week 20–22 pretraining project.
- [epoch-research/scaling-laws](https://github.com/epoch-ai/scaling-laws-decomposition) — Epoch AI's analysis and decomposition of scaling law research.

## Documentation

- [PyTorch profiler](https://pytorch.org/docs/stable/profiler.html) — For measuring real FLOP counts and verifying your 6ND estimates.

## Optional / Bonus

- [Beyond Chinchilla-Optimal](https://arxiv.org/abs/2401.00448) — Gadre et al. 2024. Argues for even more data than Chinchilla recommends for practical deployment.
- [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760) — OpenAI 2022. Scaling laws extended to RLHF reward models.
- [Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling](https://arxiv.org/abs/2304.01373) — EleutherAI 2023. Models at many scales with checkpoints; useful for empirically studying scaling.
