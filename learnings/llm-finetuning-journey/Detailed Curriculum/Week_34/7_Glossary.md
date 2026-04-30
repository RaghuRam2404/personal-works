# Week 34 Glossary

**Unsloth**: An open-source library (github.com/unslothai/unsloth) providing optimized Triton/CUDA kernels for LoRA/QLoRA training; achieves 2–5x speedup over vanilla HuggingFace training.

**FastLanguageModel**: Unsloth's main class; provides `from_pretrained` (loads model with Unsloth kernels) and `get_peft_model` (applies optimized LoRA adapters).

**Flash Attention 2**: An attention algorithm by Dao et al. that computes attention in O(n) memory (vs. O(n²) for standard attention) using block tiling; reduces VRAM for long sequences.

**Triton kernel**: A Python-like language for writing custom GPU kernels; used by Unsloth to implement fused LoRA and RoPE operations with better performance than PyTorch defaults.

**Fused kernel**: A GPU kernel that combines multiple operations (e.g., two matrix multiplies) into one pass, reducing memory bandwidth overhead and improving compute utilization.

**RoPE (Rotary Position Embedding)**: Position encoding method used by Qwen2.5, Llama, and Mistral; encodes position by rotating query and key vectors in attention. Unsloth has a custom kernel to precompute and cache these rotations.

**`use_gradient_checkpointing="unsloth"`**: Unsloth's custom gradient checkpointing implementation, passed to `FastLanguageModel.get_peft_model()`; more memory-efficient than PyTorch's default.

**Kernel fusion**: Combining multiple GPU operations into a single kernel launch, reducing kernel launch overhead and GPU-CPU synchronization; a key optimization technique in Unsloth.

**Packing ratio**: The average number of training examples packed into one sequence at `max_seq_length`; higher ratio = better GPU utilization. Typical for SQL examples at seq_len 512: 2–4x.

**Steps per second**: The throughput metric for training; measures how many optimizer update steps complete per second. The primary measure for comparing training frameworks.
