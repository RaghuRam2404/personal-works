# Week 3 — Resources

## Videos

- [Building makemore Part 4: Becoming a Backprop Ninja](https://www.youtube.com/watch?v=q8SA3rM6ckI) — Andrej Karpathy, YouTube, 1h55m. **Critical. Code along. Every line.**
- [Building makemore Part 5: Building a WaveNet](https://www.youtube.com/watch?v=t3YJ5hKiMQ0) — Andrej Karpathy, YouTube, 1h56m. Hierarchical character-level LM using dilated convolutions.
- [But what is a convolution?](https://www.youtube.com/watch?v=KuXjwB4LzSA) — 3Blue1Brown, YouTube, 23m. Outstanding visual intuition for the convolution operation.
- [Convolutional Neural Networks Explained](https://www.youtube.com/watch?v=YRhxdVk_sIs) — StatQuest, YouTube, 23m. Accessible walkthrough before diving into CS231n.

## Blog Posts / Articles

- [CS231n Convolutional Neural Networks for Visual Recognition — ConvNet Architecture](https://cs231n.github.io/convolutional-networks/) — Karpathy's original course notes. **Required reading.** Cover the convolution, pooling, and normalization sections.
- [An Intuitive Explanation of Convolutional Neural Networks](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/) — Ujjwal Karn. Good complementary read to the CS231n notes.
- [Computing Receptive Fields of CNNs](https://distill.pub/2019/computing-receptive-fields/) — Araujo et al., Distill. Interactive receptive field calculator.

## Papers

- [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499) — van den Oord et al., 2016. Read sections 1–2 (introduction and dilated causal convolutions). The architecture diagram in Figure 3 is essential.
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) — He et al., 2015. Read the abstract and Section 2. You will implement ResNets in Phase 2, but understanding why residuals exist starts here.

## Documentation

- [nn.Conv2d docs](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) — Note the `dilation` parameter and the output size formula.
- [nn.BatchNorm2d docs](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html) — Differs from `BatchNorm1d` in which dims are normalized.
- [torchvision.transforms docs](https://pytorch.org/vision/stable/transforms.html) — Reference for `RandomCrop`, `RandomHorizontalFlip`, `Normalize`.
- [torchvision.datasets.CIFAR10 docs](https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html) — One-liner dataset loading.

## GitHub Repos

- [karpathy/nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero) — Lectures 4 and 5 in the makemore series live here.
- [kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar) — Clean PyTorch CIFAR-10 training scripts; ResNet, VGG, and DenseNet implementations on CIFAR-10.

## Optional / Bonus

- [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285) — Dumoulin & Visin. The definitive technical reference for every convolution variant (transposed, dilated, grouped). Sections 1–4 are most relevant now.
- [Visualizing and Understanding Convolutional Networks](https://arxiv.org/abs/1311.2901) — Zeiler & Fergus, 2014. The paper that first visualized what CNN filters learn. Sobering and important.
