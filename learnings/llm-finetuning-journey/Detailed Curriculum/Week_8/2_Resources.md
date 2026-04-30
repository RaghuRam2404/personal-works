# Week 8 Resources — Phase 1 Gate / Capstone

## Papers

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017. Skim the abstract and Figure 1 this week. You will study this fully in Phase 2.
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — Kaplan et al., 2020. Establishes the relationship between model size, data, compute, and loss. Required reading in Phase 3 but a worthwhile preview now.

## Videos

- [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Andrej Karpathy — 1h56m. Preview for Phase 2. Watch the first 30 minutes this week as context for Option A — the full video is assigned in Week 11.

## Blog Posts / Articles

- [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) — Andrej Karpathy. Re-read for the third time. After 8 weeks of hands-on training, you will understand it differently now.
- [Reproducibility, Baselines, and Training Stability in Deep Learning](https://www.cs.toronto.edu/~rgrosse/courses/csc421_2019/readings/L11%20Training%20Tips.pdf) — Roger Grosse. A rigorous checklist-style guide to training discipline; useful for the capstone self-assessment.
- [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) — Rich Sutton, 2019. A 1-page essay arguing that general methods + compute beat domain-specific methods. Relevant motivation for this entire 18-month course.
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Jay Alammar. The best visual introduction to the transformer architecture. Read Section 1 (encoder) and Section 2 (decoder) over the weekend before Week 9.

## GitHub Repos

- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) — The reference architecture for Option A. Study `model.py` (the GPT class) and `train.py` for the training loop pattern. Do NOT copy-paste — type everything yourself.
- [wandb/wandb](https://github.com/wandb/wandb) — Weights & Biases Python client. Source for understanding how `wandb.init()` and `wandb.log()` work under the hood.

## Documentation

- [HuggingFace Trainer documentation](https://huggingface.co/docs/transformers/main_classes/trainer) — If you choose Option B with `Trainer`. Read the `TrainingArguments` parameter list — every argument corresponds to something you implemented manually in earlier weeks.
- [torch.utils.data.DataLoader docs](https://pytorch.org/docs/stable/data.html) — For your custom data loader in Option A. Pay attention to `num_workers`, `pin_memory`, and `drop_last`.
- [nanoGPT README](https://github.com/karpathy/nanoGPT#readme) — Explains how to scale the architecture. Use the "baby GPT" config for your capstone (fastest training).
- [Weights & Biases Quickstart](https://docs.wandb.ai/quickstart) — If you have not used W&B from a Colab notebook before. Covers `wandb.init()`, `wandb.log()`, and the W&B report builder.
- [W&B Reports](https://docs.wandb.ai/guides/reports) — How to create a shareable comparison report for your capstone README.

## Optional / Bonus

- [nanoGPT README — baby GPT config](https://github.com/karpathy/nanoGPT#readme) — The README explains how to scale the architecture. Use the "baby GPT" config for your capstone.
