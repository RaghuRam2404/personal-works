# Week 19 TakeAway — Distributed Training

**One-liner:** DDP replicates model; ZeRO shards optimizer/gradients/params; FSDP = ZeRO-3 in PyTorch; Accelerate makes it portable.

---

## Memory Per GPU — 7B Model, 8 GPUs

| Stage | Per-GPU (GB) |
|---|---|
| DDP | 98 |
| ZeRO-1 | 49 |
| ZeRO-2 | 24.5 |
| ZeRO-3 / FSDP | 12.25 |

Formula: Params (FP16) = 2N bytes; Grads (FP32) = 4N bytes; Adam states (FP32) = 8N bytes

---

## Key Code Pattern — Accelerate

```python
from accelerate import Accelerator

accelerator = Accelerator(
    mixed_precision="fp16",
    gradient_accumulation_steps=4
)

model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

for x, y in loader:
    with accelerator.accumulate(model):
        loss = criterion(model(x), y)
        accelerator.backward(loss)  # NOT loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

---

## Decision Rules

- Model fits on 1 GPU → use single GPU; use Accelerate for mixed precision and portability
- Model fits with ZeRO-2 (shards optimizer + grads) → use ZeRO-2 before ZeRO-3
- NVLink interconnect available → ZeRO-3 communication overhead is acceptable
- Ethernet / slow interconnect → prefer ZeRO-2 to avoid all-gather bottleneck
- Need gradient accumulation without distributed → set `gradient_accumulation_steps` in Accelerator
- 50M model → single A100, no ZeRO needed; fits in < 1GB

---

## Numbers to Remember

| Fact | Value |
|---|---|
| FP32 bytes per param | 4 |
| FP16 bytes per param | 2 |
| Adam: extra bytes per param | 8 (FP32) |
| Gradient bytes per param | 4 (FP32) |
| ZeRO-2 memory factor | (2 + 4/N + 8/N) × N bytes |
| 50M model total memory | ~500MB in FP16 (trivially fits) |

---

## Red Flags

- `loss.backward()` with Accelerate → gradient scaling skipped, training may diverge in FP16
- No `accelerator.prepare()` on dataloader → data stays on CPU
- FSDP on single GPU → no benefit, extra overhead
- ZeRO-3 on slow interconnect → may be slower than ZeRO-2 even with lower memory
