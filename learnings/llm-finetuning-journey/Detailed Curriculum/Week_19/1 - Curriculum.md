# Week 19 — Distributed Training: DDP, FSDP, and ZeRO

## Learning Objectives

By the end of this week, you will be able to:

- Explain data parallelism, model parallelism, and pipeline parallelism at a conceptual level
- Describe ZeRO Stages 1, 2, and 3 and what each shards across GPUs
- Explain why FSDP superseded the original PyTorch DDP for large models
- Use HuggingFace Accelerate to run single-GPU training with minimal code changes
- Read a multi-GPU FSDP config and explain what each line does

---

## Concepts

### Why Distributed Training Exists

A single A100 80GB GPU can hold at most ~40B model parameters in FP16 (without optimizer states). A 7B model with AdamW optimizer states (2× params = 14B values) + gradients (7B values) + activations requires roughly 60–70GB — barely fits on one A100. A 70B model is impossible on a single GPU.

Distributed training splits this work across multiple GPUs. The key insight: there are three fundamentally different things you can split:
1. Data (data parallelism)
2. Model parameters (model parallelism)
3. Layers (pipeline parallelism)

Modern systems combine all three.

### Data Parallelism (DP) and Distributed Data Parallelism (DDP)

The simplest approach: replicate the full model on every GPU, split the batch, average gradients.

**PyTorch DDP (`torch.nn.parallel.DistributedDataParallel`):**
1. Each GPU holds a full copy of the model
2. Each GPU processes a different micro-batch
3. After backward pass, gradients are all-reduced (summed and averaged) across all GPUs using NCCL
4. All GPUs update with the same gradient → all replicas remain synchronized

**Memory cost of DDP:**
Every GPU stores: model weights + optimizer states + gradients + activations. Memory requirement is identical to single-GPU training. DDP gives you a throughput speedup proportional to the number of GPUs (with communication overhead), but does not help if the model does not fit on one GPU.

**When DDP is appropriate:** Model fits on one GPU; you want to process larger batches faster.

### ZeRO (Zero Redundancy Optimizer)

Paper: [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) (Rajbhandari et al., Microsoft).

DDP wastes memory because every GPU stores the full optimizer state and gradients. ZeRO partitions this redundancy across GPUs. Three stages:

| Stage | What is sharded | Memory reduction |
|---|---|---|
| ZeRO-1 | Optimizer states | ~4× |
| ZeRO-2 | Optimizer states + gradients | ~8× |
| ZeRO-3 | Optimizer states + gradients + parameters | ~64× (proportional to N_GPUs) |

**ZeRO-1:** Each GPU owns 1/N of the optimizer state. During the optimizer step, each GPU updates its shard, then all-gathers to sync parameters. Forward/backward pass is same as DDP.

**ZeRO-2:** Each GPU also owns 1/N of the gradient tensor. After backward pass, reduce-scatter is used instead of all-reduce — each GPU only accumulates the gradient for its shard.

**ZeRO-3 (FSDP-like):** Parameters themselves are sharded. Before each forward pass, each layer's parameters are all-gathered across GPUs; after the backward pass, the parameters are discarded from GPUs that don't own them. Minimal memory, maximum communication.

**The communication trade-off:** ZeRO-3 shards everything but must all-gather parameters every forward pass. On fast NVLink interconnects, this overhead is manageable. On slow Ethernet (cloud VMs), ZeRO-3 can be slower than ZeRO-2.

### FSDP (Fully Sharded Data Parallelism)

PyTorch's native implementation of ZeRO-3 semantics. Available since PyTorch 1.12. FSDP wraps model modules individually, sharding their parameters across GPUs and gathering them on demand.

Key FSDP parameters you will encounter:
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

model = FSDP(
    model,
    auto_wrap_policy=transformer_auto_wrap_policy,
    mixed_precision=MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float16,
    ),
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3
    cpu_offload=CPUOffload(offload_params=False),    # keep on GPU
)
```

**`auto_wrap_policy`:** FSDP needs to know which modules to shard. `transformer_auto_wrap_policy` wraps each transformer block independently, enabling efficient all-gather per layer.

**`mixed_precision`:** Parameters in FP16, gradient reduction in FP32 (avoids underflow), buffers in FP16.

### Pipeline Parallelism

Different layers on different GPUs, inputs flowing through in a pipeline. GPT-3's training used pipeline parallelism (Megatron-LM). You will not implement this — but knowing it exists helps you understand why Llama 3's training required thousands of GPUs.

### HuggingFace Accelerate — Practical Single/Multi-GPU Training

Accelerate is HuggingFace's abstraction layer that wraps DDP/FSDP/DeepSpeed with a single API. For your purposes (single Colab GPU), Accelerate makes your training script portable without restructuring it.

**Minimal Accelerate setup:**

```python
from accelerate import Accelerator

accelerator = Accelerator(mixed_precision="fp16")

model, optimizer, train_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader
)

for batch in train_dataloader:
    with accelerator.accumulate(model):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
```

**Gradient accumulation with Accelerate:**

```python
accelerator = Accelerator(
    gradient_accumulation_steps=4,  # effective batch = 4 × batch_size
    mixed_precision="fp16"
)
```

**The key value proposition for you:** When you move to RunPod (multi-GPU) in a later phase, the same code runs without changes. Accelerate handles the DDP/FSDP setup automatically based on the number of available GPUs.

**Accelerate config file (`accelerate_config.yaml`):**

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: NO     # for single GPU
mixed_precision: fp16
num_processes: 1
num_machines: 1
```

Change `distributed_type: MULTI_GPU` and `num_processes: 8` to go multi-GPU.

### Model Parallelism (Tensor Parallelism)

Megatron-LM's approach: split individual weight matrices across GPUs. The attention heads in GPT-3 are split across 96 GPUs so each GPU holds a fraction of each attention layer. This requires modifying the model code itself (unlike FSDP, which is transparent to the model).

You will not implement this, but you must understand it to read papers about Llama 3 and DeepSeek training infrastructure.

### Why You Do Not Need Multi-GPU This Phase

Your Week 20–22 project trains a 50M-parameter model. 50M params in FP32 = 200MB. With optimizer states and activations, call it 1–2GB — trivially fits on any A100 or even a T4 (16GB). Single-GPU training is correct for this phase.

What this week teaches you conceptually prepares you for Phase 5–6, when you will fine-tune 7B models that require careful memory management.

---

## Connections

**Prior week (18):** Data pipeline produces tokens. This week covers how those tokens are consumed efficiently during training.

**Weeks 20–22:** You will use HuggingFace Accelerate directly. The `accelerator.prepare()` pattern is what you will implement.

**Phase 4 (Week 28+):** Full SFT and LoRA on 7B models will require understanding mixed precision and gradient checkpointing, both touched on this week.

---

## Common Misconceptions

- **"I need multiple GPUs to use FSDP."** FSDP works on a single GPU but provides no sharding benefit. Its value is multi-GPU.
- **"ZeRO-3 is always better than ZeRO-2."** ZeRO-3 reduces memory further but adds communication. On slow interconnects, ZeRO-2 with model offloading to CPU may outperform ZeRO-3.
- **"Gradient accumulation is the same as a larger batch."** It produces mathematically identical gradients only if BatchNorm is not used (which transformers don't use). For transformers, gradient accumulation ≡ larger batch.
- **"Accelerate automatically chooses the best distributed strategy."** Accelerate uses whatever you configure. You must explicitly choose DDP vs. FSDP vs. DeepSpeed ZeRO in the config.

---

## Time Allocation (6–8 hrs)

- 1h: Read PyTorch DDP tutorial (understand the `init_process_group` and `all_reduce` concepts)
- 1.5h: Read ZeRO paper (Sections 1–4; focus on the memory analysis table in Section 2)
- 1h: Read HuggingFace Accelerate concept guides
- 2h: Coding — integrate Accelerate into your nanoGPT prototype and run it on Colab GPU
- 1h: Read the FSDP config for a publicly available training script (e.g., Llama 3's training config) and annotate each line in your `journal.md`
- 0.5h: Commit and write `journal.md` notes
