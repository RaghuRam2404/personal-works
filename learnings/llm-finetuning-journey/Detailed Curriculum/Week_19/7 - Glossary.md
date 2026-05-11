# Week 19 Glossary — Distributed Training

**Data parallelism (DP)**: Training strategy where the full model is replicated on every device and each device processes a different data batch; gradients are averaged across devices.

**DDP (Distributed Data Parallelism)**: PyTorch's implementation of data parallelism using NCCL for gradient all-reduce; each GPU holds a full model copy.

**Model parallelism**: Splitting model layers or weight matrices across multiple devices; required when a single model does not fit on one GPU.

**Pipeline parallelism**: Assigning different transformer layers to different GPUs; data flows through stages sequentially; used in Megatron-LM for GPT-3 training.

**Tensor parallelism**: Splitting individual weight matrices across GPUs at the matrix multiplication level; requires model code modifications.

**ZeRO (Zero Redundancy Optimizer)**: A memory optimization strategy from Microsoft DeepSpeed that eliminates redundant optimizer state, gradient, and parameter storage across GPUs.

**ZeRO Stage 1**: Partitions optimizer states across GPUs; parameters and gradients remain full on each GPU.

**ZeRO Stage 2**: Partitions optimizer states and gradients across GPUs; parameters remain full on each GPU.

**ZeRO Stage 3**: Partitions optimizer states, gradients, and parameters across GPUs; parameters are all-gathered on demand.

**FSDP (Fully Sharded Data Parallelism)**: PyTorch's native ZeRO-3-equivalent implementation; wraps modules to shard and gather parameters automatically.

**All-reduce**: A collective communication operation where values from all GPUs are summed and the result is broadcast back to all GPUs.

**Reduce-scatter**: A collective operation where values are summed and the result is distributed (scattered) in shards to each GPU; used in ZeRO-2/3.

**All-gather**: A collective operation where each GPU's shard is collected and broadcast to form a complete tensor on all GPUs; used in ZeRO-3 to materialize parameters.

**NCCL (NVIDIA Collective Communications Library)**: The GPU communication library used by PyTorch DDP, FSDP, and DeepSpeed.

**Gradient accumulation**: Summing gradients over N mini-batches before performing a parameter update, equivalent to training with an N× larger effective batch.

**Accelerate (HuggingFace)**: An abstraction library that wraps DDP, FSDP, and DeepSpeed behind a single API for portable training code.

**MFU (Model FLOP Utilization)**: The fraction of peak hardware FLOP/s actually used; accounts for communication and memory bandwidth overhead in distributed training.

**CPU offload**: Technique in ZeRO-Infinity and FSDP where optimizer states or parameters are moved to CPU RAM when not needed on GPU, enabling extremely large models at the cost of speed.
