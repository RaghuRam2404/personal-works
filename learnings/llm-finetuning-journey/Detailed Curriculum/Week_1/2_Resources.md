# Week 1 — Resources

## Videos

- [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) — Andrej Karpathy, YouTube, 2h25m. **Watch fully and code along. This is the core of Week 1.**
- [PyTorch in 100 Seconds](https://www.youtube.com/watch?v=ORMx45xqWkA) — Fireship, YouTube, 2m. Good 2-minute orientation before diving in.
- [What is Backpropagation Really Doing?](https://www.youtube.com/watch?v=Ilg3gGewQ5U) — 3Blue1Brown, YouTube, 13m. Visual intuition for gradient flow.
- [Backpropagation Calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8) — 3Blue1Brown, YouTube, 10m. The math layer under the visual intuition.

## Blog Posts / Articles

- [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) — Andrej Karpathy. Read this three times across Phase 1. It will save you more debugging hours than any other resource.
- [Yes You Should Understand Backprop](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b) — Andrej Karpathy. Short. Read before watching the micrograd video.
- [PyTorch Autograd Explained](https://towardsdatascience.com/pytorch-autograd-understanding-the-heart-of-pytorchs-magic-2686cd94ec95) — Towards Data Science. A solid companion read after the video.

## Documentation

- [PyTorch Learn the Basics](https://pytorch.org/tutorials/beginner/basics/intro.html) — Official tutorial, all 8 sections. Work through Tensors, Datasets, DataLoaders, Transforms, Build Model, Autograd, Optimization, Save/Load.
- [torch.Tensor docs](https://pytorch.org/docs/stable/tensors.html) — Bookmark this. You will reference it weekly.
- [torch.autograd docs](https://pytorch.org/docs/stable/autograd.html) — Understand `backward()`, `grad_fn`, and `torch.no_grad`.
- [torch.optim docs](https://pytorch.org/docs/stable/optim.html) — Read the `SGD` and `Adam` entries in full.

## GitHub Repos

- [karpathy/micrograd](https://github.com/karpathy/micrograd) — The reference implementation. Do not look at this until after you have written your own.
- [karpathy/nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero) — All Karpathy lecture notebooks. Week 1 uses `micrograd/` folder.

## Papers

No papers this week — you are building foundations. Papers start in Week 5.

## Optional / Bonus

- [The Matrix Calculus You Need for Deep Learning](https://arxiv.org/abs/1802.01528) — Parr & Howard. If you want the full mathematical derivation of gradients for matrix operations. Sections 3–5 are most relevant.
- [PyTorch Internals](http://blog.ezyang.com/2019/05/pytorch-internals/) — Edward Yang. If you are curious how PyTorch implements autograd under the hood in C++. Not required this week.
