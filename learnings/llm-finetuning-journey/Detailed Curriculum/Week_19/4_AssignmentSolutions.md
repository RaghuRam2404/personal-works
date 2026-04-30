# Week 19 Assignment Solutions

## Task 1 — Key Snippet: Accelerate Integration

```python
from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader, TensorDataset

accelerator = Accelerator(
    mixed_precision="fp16",
    gradient_accumulation_steps=4
)

model = torch.nn.Linear(128, 128)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
dataset = TensorDataset(torch.randn(1000, 128), torch.randn(1000, 128))
loader = DataLoader(dataset, batch_size=16)

model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
print(f"Device: {accelerator.device}")  # cuda:0 on Colab

for epoch in range(3):
    for x, y in loader:
        with accelerator.accumulate(model):
            pred = model(x)
            loss = ((pred - y)**2).mean()
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
    accelerator.log({"loss": loss.item(), "epoch": epoch})
    print(f"Epoch {epoch}, loss: {loss.item():.4f}")
```

**Common gotchas:**
- Forgetting `accelerator.accumulate(model)` context manager — gradient accumulation won't work
- Calling `loss.backward()` instead of `accelerator.backward(loss)` — mixed precision gradient scaling will be skipped
- Not calling `accelerator.prepare()` on the dataloader — data won't be moved to the correct device
- `mixed_precision="fp16"` on CPU (Colab CPU session) will throw an error; use `"no"` when on CPU

---

## Task 3 — ZeRO Memory Analysis (7B model, 8 GPUs)

**Memory per component (full precision mixed training):**
- Params (FP16): 7e9 × 2 bytes = 14 GB
- Gradients (FP32): 7e9 × 4 bytes = 28 GB
- Optimizer states (FP32, Adam = 2× params): 7e9 × 8 bytes = 56 GB
- Total on single GPU: 14 + 28 + 56 = **98 GB**

| Stage | What is sharded | Per-GPU memory (GB) |
|---|---|---|
| Single GPU / DDP | Nothing | 14 + 28 + 56 = **98 GB** |
| ZeRO-1 | Optimizer states ÷ 8 | 14 + 28 + 7 = **49 GB** |
| ZeRO-2 | Optimizer + grads ÷ 8 | 14 + 3.5 + 7 = **24.5 GB** |
| ZeRO-3 | Everything ÷ 8 | 1.75 + 3.5 + 7 = **12.25 GB** |

**Key insight:** A 7B model fits on a single A100 80GB under DDP (98GB > 80GB — doesn't fit!). Under ZeRO-2 it fits on an A100 (24.5GB). Under ZeRO-3 it fits on a 24GB consumer GPU (12.25GB + activations).

This is why QLoRA is so popular: it further reduces the 14GB param cost via 4-bit quantization, making 7B models fit on a 16GB GPU even without ZeRO.

**Common gotchas:**
- Forgetting activation memory (can be 2–10× params for long sequences without gradient checkpointing)
- Confusing FP16 inference memory vs. FP32 training memory — training needs more
- ZeRO-3 numbers above ignore the all-gather communication buffer, which adds ~1 layer's parameters per device

---

## Task 4 — How to Verify You Did It Right

**Q1 (7B on 4 A100s — which ZeRO stage?):**

Best answer: ZeRO-2. Reasoning:
- ZeRO-2 brings per-GPU memory to 24.5GB — fits on A100 80GB with room for activations
- ZeRO-3 is overkill (80GB A100 can hold the model under ZeRO-2)
- ZeRO-3 adds communication overhead that hurts throughput on 4 GPUs connected via NVLink
- If you wanted to fine-tune 70B on 4 A100s, ZeRO-3 would be necessary

**Q3 (Why FSDP doesn't need model changes):**

FSDP operates at the module boundary level — it wraps existing `nn.Module` objects and handles parameter gathering transparently. The model's `forward()` method is called identically; FSDP intercepts the parameter access underneath. Tensor parallelism requires the linear layer's weight matrix to be split at the matrix multiplication level, which requires modifying the `forward()` computation itself (partial matrix multiply → all-reduce → continue). FSDP is transparent; tensor parallelism is intrusive.
