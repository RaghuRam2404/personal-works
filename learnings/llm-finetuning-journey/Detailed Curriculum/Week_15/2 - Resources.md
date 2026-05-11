# Week 15 Resources

## Papers

- [Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — Radford et al. 2019. Re-read Section 2 (model architecture) this week. Note the Pre-LN and GELU details.
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Dao et al. 2022. Read the introduction and Figure 1 to understand the IO bottleneck. The rest is optional.
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Dao et al. 2023. The version used in production today.

## Videos

- [Andrej Karpathy — Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU) — 4h01m. The primary resource. Code along. Every minute is worth it.

## Blog Posts / Articles

- [Training Neural Networks — Andrej Karpathy](https://karpathy.github.io/2019/04/25/recipe/) — Karpathy's training recipe. The debugging philosophy behind this week's project.
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html) — PyTorch official AMP documentation. Read "Autocasting" and "Gradient Scaling" sections.

## GitHub Repos

- [karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt) — The repo for the 4-hour GPT-2 video. The final `train_gpt2.py` is the reference. Do not look at it until you've completed your own implementation.
- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) — The cleaned-up, more production-ready version. Read `train.py` after completing your own.
- [openai-community/gpt2 on HuggingFace](https://huggingface.co/openai-community/gpt2) — The pretrained GPT-2 124M weights. Use for weight loading verification.

## Documentation

- [torch.autocast](https://pytorch.org/docs/stable/amp.html#torch.autocast) — The autocast context manager. Pay attention to which operations are autocasted and which stay in FP32.
- [torch.nn.functional.scaled_dot_product_attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) — The Flash Attention drop-in. Understand `is_causal`, `attn_mask`, `dropout_p` parameters.
- [HellaSwag dataset](https://huggingface.co/datasets/Rowan/hellaswag) — The evaluation dataset. Download and examine its format before implementing the evaluator.

## Optional / Bonus

- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — Kaplan et al. 2020. Preview for Week 17. Understanding where GPT-2 124M sits on the scaling curve.
- [torch.compile documentation](https://pytorch.org/docs/stable/generated/torch.compile.html) — Stretch goal: compiling your model for 10–30% additional speedup.
- [FineWeb-Edu dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) — Alternative to OpenWebText. Higher quality educational content. Good for training a model you'll later fine-tune for SQL.
