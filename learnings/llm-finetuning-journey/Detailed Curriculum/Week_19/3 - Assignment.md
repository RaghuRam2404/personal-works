# Week 19 Assignment — Distributed Training Concepts + Accelerate Hands-On

## Setup Checklist

- [ ] `pip install accelerate` (latest)
- [ ] `accelerate config` completed (choose "No distributed training" for Colab single GPU)
- [ ] A minimal nanoGPT training script (from Week 20 preview, or download from [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT))
- [ ] GitHub repo with `week-19-distributed-concepts/` directory

---

## Task 1 — Integrate HuggingFace Accelerate into a Minimal Training Loop

**Goal:** Convert a plain PyTorch training loop to use Accelerate without changing the core logic.

**Requirements:**

Start from this minimal training loop:

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

# Toy model and data
model = torch.nn.Linear(128, 128)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
dataset = TensorDataset(torch.randn(1000, 128), torch.randn(1000, 128))
loader = DataLoader(dataset, batch_size=16)

for epoch in range(3):
    for x, y in loader:
        pred = model(x)
        loss = ((pred - y)**2).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Epoch {epoch}, loss: {loss.item():.4f}")
```

Modify it to:
- Use `Accelerator` with `mixed_precision="fp16"` (or `"no"` if on CPU)
- Use `accelerator.prepare()` on model, optimizer, and dataloader
- Use `accelerator.backward(loss)` instead of `loss.backward()`
- Use `accelerator.log({"loss": loss.item()})` for logging
- Add `gradient_accumulation_steps=4`
- Print device placement information: `print(accelerator.device)`

**Deliverable:** `week-19-distributed-concepts/accelerate_minimal.py`

**Acceptance criteria:** Script runs without error on Colab single GPU. Loss decreases over 3 epochs. Mixed precision is active (check via `accelerator.mixed_precision`).

---

## Task 2 — Annotate an FSDP Config

**Goal:** Read and understand a real multi-GPU FSDP configuration.

**Requirements:**

Find any public FSDP training config — for example:
- [torchtune Llama3 FSDP recipe](https://github.com/pytorch/torchtune/blob/main/recipes/full_finetune_distributed.py)
- OR create your own hypothetical config for a 7B model on 8 A100s

Write `week-19-distributed-concepts/fsdp_config_annotated.md` with a copy of the config (or a 20-line excerpt) where every meaningful line has an inline comment explaining what it does and why.

You must explain these specific concepts in your annotations:
- What `ShardingStrategy.FULL_SHARD` means (vs `SHARD_GRAD_OP`)
- What `auto_wrap_policy` does and why it is needed
- What `mixed_precision` settings mean (param_dtype vs reduce_dtype)
- What `cpu_offload` does and when you would use it

**Deliverable:** `fsdp_config_annotated.md`

---

## Task 3 — ZeRO Memory Analysis

**Goal:** Compute the per-GPU memory requirements for a model under each ZeRO stage.

**Requirements:**

For a hypothetical model with N = 7B parameters, Adam optimizer (2 optimizer states per param), and 8 GPUs:

Compute per-GPU memory (in GB) for:
- No parallelism (single GPU): params + grad + optimizer states, all in FP32
- DDP: same as single GPU (no reduction in memory)
- ZeRO-1 (optimizer state sharding only)
- ZeRO-2 (optimizer states + gradients sharded)
- ZeRO-3 (all sharded)

Assumptions:
- FP32: 4 bytes per value; FP16: 2 bytes per value
- Params stored in FP16: 2 × N bytes
- Gradients in FP32: 4 × N bytes
- Adam optimizer states (momentum + variance): 2 × 4 × N bytes in FP32
- Activations: ignore for this calculation (assume gradient checkpointing eliminates them)

Present as a Markdown table in `week-19-distributed-concepts/zero_memory_analysis.md`.

**Stretch:** Also compute the per-GPU memory for a 70B model on 16 GPUs under each ZeRO stage.

---

## Task 4 — journal.md Entry

**Goal:** Solidify conceptual understanding by writing it in your own words.

**Requirements:**

Write a 400-word journal entry in `journal.md` answering:
1. If you had access to 4 A100 80GB GPUs today, which ZeRO stage would you use to fine-tune Qwen2.5-7B, and why?
2. What is the difference between gradient accumulation and data parallelism? When would you use each?
3. Why does tensor parallelism require model code changes while FSDP does not?

**Deliverable:** `week-19-distributed-concepts/journal.md`. GitHub commit `week-19-distributed-concepts`.

---

## Stretch Goals

- Measure actual GPU memory usage of your Accelerate training loop using `torch.cuda.memory_allocated()` and `torch.cuda.max_memory_allocated()` before and after `accelerate.prepare()`
- Run `accelerate launch --num_processes 2` on your script if you have a machine with 2 GPUs (e.g., an older machine or RunPod 2×A6000 for <$1)
- Read the [DeepSpeed ZeRO-Infinity paper](https://arxiv.org/abs/2104.07857) and note how it extends ZeRO-3 to NVMe storage
