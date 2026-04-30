# Week 34 Resources

## Papers

- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Dao 2023. The attention optimization that Unsloth integrates. Read the abstract and Section 3 for the tiling algorithm.

## Videos

- [Unsloth: 2x Faster, 70% Less VRAM — Daniel Han (GPU MODE)](https://www.youtube.com/watch?v=Cz7jQvGfqS0) — Daniel Han (Unsloth) — ~35 min. GPU MODE conference presentation covering the Triton kernel optimizations, QLoRA integration, and benchmark results.
- [Fine-tuning LLMs with Unsloth and TRL](https://www.youtube.com/watch?v=aQmoog_s8_k) — Unsloth AI — ~20 min. Step-by-step walkthrough of the Unsloth + TRL SFTTrainer pipeline; complements the Qwen2.5 notebook.

## Blog Posts / Articles

- [Unsloth Blog](https://unsloth.ai/blog) — Read the 3 most recent posts. The Mistral and Llama benchmark posts contain detailed performance comparisons.
- [Unsloth README](https://github.com/unslothai/unsloth/blob/main/README.md) — The official README contains up-to-date installation instructions and benchmark numbers. Required reading.

## GitHub Repos

- [Unsloth](https://github.com/unslothai/unsloth) — Main repo. Look at `unsloth/kernels/` for Triton kernel implementations and `unsloth/models/` for model-specific patches.
- [Unsloth notebooks](https://github.com/unslothai/unsloth/tree/main/notebooks) — Pre-made Colab notebooks for Qwen2.5, Llama 3, and other models; good starting points for your training setup.

## Documentation

- [Unsloth documentation](https://docs.unsloth.ai/) — Official docs with quickstart guides and API reference.
- [TRL SFTTrainer compatibility note](https://huggingface.co/docs/trl/sft_trainer) — Unsloth models are compatible with standard TRL `SFTTrainer`.

## Optional / Bonus

- [Liger Kernel](https://github.com/linkedin/Liger-Kernel) — LinkedIn's alternative to Unsloth's approach; provides fused kernels that work with vanilla PyTorch/HF. Relevant context for understanding the kernel optimization ecosystem.
- [torch.compile for training](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) — PyTorch 2.0's alternative approach to kernel fusion via tracing and compilation; different from Unsloth's custom Triton kernels but worth knowing.
