# Week 20 Resources — Pretraining Setup

## Papers

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al. 2017. Review the GPT architecture differences: causal attention, pre-LN.
- [Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — Radford et al. 2019. Section 2 describes the model architecture you are implementing.
- [Cramming: Training a Language Model on a Single GPU in One Day](https://arxiv.org/abs/2212.14034) — Geiping & Goldstein 2022. What choices matter most in a budget training run.
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135) — Dao et al. 2022. Why `F.scaled_dot_product_attention` is an important optimization.

## Videos

- [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Andrej Karpathy (~2h). The definitive implementation walkthrough. Watch the whole thing.
- [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU) — Andrej Karpathy (~4h). Focus on the data loading, training loop, and torch.compile sections.

## Blog Posts / Articles

- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) — Jay Alammar. Visual walkthrough of the GPT-2 architecture.
- [nanogpt-lecture](https://github.com/karpathy/ng-video-lecture) — Karpathy's lecture notebook. Reference implementation you will learn from but rewrite.
- [HuggingFace Tokenizers Quick Tour](https://huggingface.co/docs/tokenizers/quicktour) — Required reading before training your BPE tokenizer.

## GitHub Repos

- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) — Primary reference. Study `model.py` and `train.py`. Do not copy-paste — rewrite from understanding.
- [karpathy/llm.c](https://github.com/karpathy/llm.c) — C implementation of GPT-2 training; useful for understanding what the GPU is actually doing.
- [huggingface/tokenizers](https://github.com/huggingface/tokenizers) — The Rust-backed tokenizers library you use in this week's assignment.

## Documentation

- [PyTorch scaled_dot_product_attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) — API reference for flash attention in PyTorch.
- [Weights & Biases Quickstart](https://docs.wandb.ai/quickstart) — Set up W&B logging for your training run (required for Week 21).
- [numpy.memmap documentation](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html) — How to read and write memory-mapped arrays.

## Optional / Bonus

- [GPT-2 reproduction guide](https://github.com/karpathy/llm.c/discussions/677) — Karpathy's detailed notes on reproducing GPT-2. Includes hyperparameter rationale.
- [Mistral-7B Technical Report](https://arxiv.org/abs/2310.06825) — Short paper; shows how a small team (15 people) trains a SOTA model. Section 2 covers architecture decisions relevant to your implementation.
- [Rotary Position Embeddings (RoPE)](https://arxiv.org/abs/2104.09864) — Su et al. 2021. Optional reading: understanding why modern LLMs moved from absolute to rotary embeddings.
