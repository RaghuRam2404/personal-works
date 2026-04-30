# Week 2 — Resources

## Videos

- [The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo) — Andrej Karpathy, YouTube, 1h15m. Bigram and simple MLP language model on names dataset.
- [Building makemore Part 2: MLP](https://www.youtube.com/watch?v=TCH_1BHY58I) — Andrej Karpathy, YouTube, 1h15m. Embedding-based MLP with proper train/val split.
- [Building makemore Part 3: Activations & Gradients, BatchNorm](https://www.youtube.com/watch?v=P6sfmUTpUmc) — Andrej Karpathy, YouTube, 1h55m. **The most important video of Week 2.** Deep dive into activation distributions, gradient flow, Kaiming init, and batch norm.

## Blog Posts / Articles

- [Yes you should understand backprop](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b) — Andrej Karpathy, Medium. Read before starting the makemore videos.
- [Batch Normalization Explained](https://towardsdatascience.com/batch-normalization-explained-algorithm-breakdown-23d2794511c) — A clear walkthrough of the math and the train/eval distinction.
- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html) — Srivastava et al., JMLR 2014. The original dropout paper; the abstract + Section 1 is sufficient.

## Papers

- [Batch Normalization: Accelerating Deep Network Training](https://arxiv.org/abs/1502.03167) — Ioffe & Szegedy, 2015. Read the abstract and algorithm box. The train/eval behavior is specified in Section 3.
- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852) — He et al., 2015. The Kaiming init paper. Read Section 2.2 for the derivation.
- [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415) — Hendrycks & Gimpel. Short paper explaining GELU. Read the abstract + Figure 1.

## Documentation

- [nn.Linear docs](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) — Note the default init (Kaiming Uniform) and the weight shape convention `(out_features, in_features)`.
- [nn.BatchNorm1d docs](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html) — Pay attention to the `training` flag and `track_running_stats`.
- [nn.Dropout docs](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html) — Behavior during training vs. eval mode.
- [torch.nn.init docs](https://pytorch.org/docs/stable/nn.init.html) — All initialization functions: `kaiming_normal_`, `xavier_uniform_`, `zeros_`, `ones_`.

## GitHub Repos

- [karpathy/nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero) — Lecture notebooks. Week 2 uses the `makemore/` folder (lectures 2–4).
- [taoyds/spider](https://github.com/taoyds/spider) — Spider text-to-SQL dataset. Download `train_spider.json` for the SQL keyword extraction task.

## Books

- [Deep Learning Book, Chapter 6: Deep Feedforward Networks](https://www.deeplearningbook.org/contents/mlp.html) — Goodfellow, Bengio, Courville. Free online. Focus on sections 6.2 (activation functions) and 6.4 (architecture design).

## Optional / Bonus

- [Layer Normalization](https://arxiv.org/abs/1607.06450) — Ba et al., 2016. The paper introducing layer norm. Important for Week 9+ when you start building transformers.
- [Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) — Glorot & Bengio, 2010. The Xavier init paper. Read Section 4 for the derivation.
