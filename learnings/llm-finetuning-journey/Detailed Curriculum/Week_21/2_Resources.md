# Week 21 Resources — Running Pretraining

## Papers

- [Cramming: Training a Language Model on a Single GPU in One Day](https://arxiv.org/abs/2212.14034) — Geiping & Goldstein 2022. Required reading for understanding budget training decisions.
- [MuP: Maximal Update Parametrization](https://arxiv.org/abs/2203.03466) — Yang et al. 2022. Why hyperparameters don't transfer across model sizes; context for your hyperparameter choices.

## Videos

- [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU) — Andrej Karpathy — 4h01m. The training loop sections are directly applicable to this week.
- [How I'd learn to train LLMs if I were starting today](https://www.youtube.com/watch?v=yBL7J0kgldU) — Yannic Kilcher, short (~15 min). Practical perspective on training budgets.

## Blog Posts / Articles

- [Andrej Karpathy's nanoGPT README](https://github.com/karpathy/nanoGPT#readme) — The training instructions and expected loss curves are directly applicable.
- [GPT-2 reproduction notes (llm.c discussion)](https://github.com/karpathy/llm.c/discussions/677) — Karpathy's notes on what to expect during a GPT-2-scale training run.
- [Loss spikes and how to handle them](https://huggingface.co/docs/transformers/v4.28.0/en/perf_train_gpu_one#gradient-clipping) — HuggingFace guide on gradient clipping and training stability.
- [Weights & Biases Experiment Tracking](https://docs.wandb.ai/guides/track) — Full guide to logging metrics, media, and artifacts to W&B.

## GitHub Repos

- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) — Reference implementation. The `train.py` in nanoGPT is your template; understand each line.
- [huggingface/accelerate](https://github.com/huggingface/accelerate) — BF16 and gradient accumulation documentation.

## Documentation

- [Weights & Biases Quickstart](https://docs.wandb.ai/quickstart) — Set up experiment tracking in 5 minutes.
- [PyTorch checkpoint saving/loading](https://pytorch.org/tutorials/beginner/saving_loading_models.html) — The definitive reference for checkpoint management.
- [HuggingFace Hub model upload](https://huggingface.co/docs/hub/models-uploading) — How to push your checkpoint to the Hub for Week 22 evaluation.

## Optional / Bonus

- [Training Neural Networks: A Road to Hell](https://www.ruder.io/neural-networks-learning-tips/) — Sebastian Ruder. Classic blog post on training instabilities and debugging.
- [Gradient Descent Variants](https://ruder.io/optimizing-gradient-descent/) — Ruder's comprehensive overview; useful background for understanding why AdamW beta2=0.95 is different from the default 0.999.
- [BF16: The Better Half Precision](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus) — Google TPU team. Why BF16's dynamic range matters for training stability.
