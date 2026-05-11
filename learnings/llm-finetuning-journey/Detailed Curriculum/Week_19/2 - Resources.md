# Week 19 Resources — Distributed Training

## Papers

- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) — Rajbhandari et al. 2020, Microsoft. The foundational paper for memory-efficient distributed training. Read Sections 1–4.
- [ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning](https://arxiv.org/abs/2104.07857) — Rajbhandari et al. 2021. Extends ZeRO to NVMe storage; optional reading.
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) — Shoeybi et al. 2019, NVIDIA. The tensor parallelism approach used for GPT-3-scale training.
- [Efficient Large Scale Language Modeling with Mixtures of Experts](https://arxiv.org/abs/2112.10684) — Artetxe et al. 2021, Meta. Shows how MoE models distribute computation; background for DeepSeek.

## Videos

- [Andrej Karpathy — Reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU) — Andrej Karpathy — ~4h. Watch the distributed training section for `torch.compile` and DDP setup; Karpathy walks through the full training loop including gradient accumulation and mixed precision.
- [PyTorch FSDP Tutorial](https://www.youtube.com/watch?v=By_O0k102PY) — PyTorch official (~30 min). Covers FSDP API and migration from DDP.

## Blog Posts / Articles

- [HuggingFace Accelerate: Concept Guides](https://huggingface.co/docs/accelerate/concept_guides/big_model_inference) — Required reading. Explains big model inference and training strategies.
- [Demystifying Parallel and Distributed Deep Learning](https://arxiv.org/abs/1802.09941) — Ben-Nun & Hoefler 2018. Academic survey of all parallelism strategies; heavy but comprehensive.
- [Microsoft DeepSpeed ZeRO Blog](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/) — Original ZeRO announcement with intuitive diagrams.
- [Lilian Weng — Large Batch Training](https://lilianweng.github.io/posts/2021-09-25-train-large/) — Weng's blog, clear diagrams for all parallelism types.

## GitHub Repos

- [huggingface/accelerate](https://github.com/huggingface/accelerate) — Required for this week's assignment.
- [microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed) — DeepSpeed ZeRO implementation; reference for production distributed training.
- [pytorch/torchtune](https://github.com/pytorch/torchtune) — PyTorch's fine-tuning library; contains production FSDP recipes for Llama and other models.
- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) — The DDP-enabled training script you will study and modify in Weeks 20–22.

## Documentation

- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) — Required reading. Understand `init_process_group`, `DistributedSampler`, and `all_reduce`.
- [PyTorch FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) — Walk through the FSDP API step by step.
- [HuggingFace Accelerate Quickstart](https://huggingface.co/docs/accelerate/quickstart) — Start here for the assignment.

## Optional / Bonus

- [Collective Communication Patterns](https://pytorch.org/tutorials/intermediate/dist_tuto.html) — PyTorch tutorial on all-reduce, broadcast, gather. Builds intuition for what happens inside DDP.
- [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965) — Google Brain 2018. The foundational pipeline parallelism paper.
