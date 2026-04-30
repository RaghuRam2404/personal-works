# Week 5 — Resources

## Videos

- [AdamW: Decoupled Weight Decay Regularization Explained](https://www.youtube.com/watch?v=oWZbcq_figk) — Yannic Kilcher, YouTube, ~25m. Clear explanation of why Adam's weight decay is broken and what AdamW fixes.
- [Learning Rate Scheduling Explained](https://www.youtube.com/watch?v=yJhto0mPrdQ) — Weights & Biases, YouTube, 20m. Overview of warmup, cosine, and step decay with W&B visualization.

## Blog Posts / Articles

- [An overview of gradient descent optimization algorithms](https://www.ruder.io/optimizing-gradient-descent/) — Sebastian Ruder. **Required reading.** The canonical reference for all first and second-order methods: SGD, Momentum, Adam, AdaGrad, RMSProp. Read all sections.
- [The Marginal Value of Momentum for Small Learning Rate SGD](https://arxiv.org/abs/1704.04861) — Sutton, 1992 (revisited). Why momentum helps more in some regimes than others.
- [Fixing Weight Decay Regularization in Adam](https://arxiv.org/abs/1711.05101) — Loshchilov & Hutter. The AdamW paper. Read the abstract, Section 2 (the bug), and Algorithm 2 (the fix).
- [Mixed Precision Training](https://developer.nvidia.com/blog/mixed-precision-training-deep-neural-networks/) — NVIDIA Developer Blog. Practical guide to AMP with worked examples.

## Papers

- [Decoupled Weight Decay Regularization (AdamW)](https://arxiv.org/abs/1711.05101) — Loshchilov & Hutter, 2019. Required. Read Sections 1–3.
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — Kingma & Ba, 2015. The original Adam paper. Read the algorithm box in Section 2.
- [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186) — Smith, 2017. Introduces the LR range test and 1-cycle policy. Relevant for the stretch goal.

## Documentation

- [torch.optim.AdamW docs](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) — Note the `amsgrad` option and the weight_decay implementation.
- [torch.cuda.amp docs](https://pytorch.org/docs/stable/amp.html) — Full API for `autocast`, `GradScaler`, and BF16 support.
- [torch.optim.lr_scheduler docs](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) — `CosineAnnealingLR`, `LinearLR`, `SequentialLR` for building warmup + cosine without a custom function.
- [torch.nn.utils.clip_grad_norm_](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html) — Returns the total norm before clipping; useful for logging.

## GitHub Repos

- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) — Study `train.py` specifically. It implements AdamW, cosine schedule with warmup, gradient clipping, and AMP (BF16) in a clean ~300-line training script. This is the gold standard for a minimal modern training loop.

## Optional / Bonus

- [Sharpness-Aware Minimization (SAM)](https://arxiv.org/abs/2010.01412) — Foret et al., 2021. Optimizer that explicitly seeks flat minima. Interesting research direction if you want to understand the geometry of loss landscapes.
- [Lion: Evolved Optimizer](https://arxiv.org/abs/2302.06675) — Chen et al., 2023. Sign-based optimizer that is simpler than Adam and uses less memory. Good read for the stretch goal.
