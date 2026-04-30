# Week 19 Quiz Answers

## Q1 — Answer: B

**Answer:** B — A full copy of the model parameters.

**Why:** DDP replicates the entire model on every GPU. Each GPU independently does a forward and backward pass on its own data slice, then all-reduces gradients to synchronize. Memory is not saved — throughput is increased.

**Why others are wrong:**
- A describes ZeRO-3/FSDP
- C describes a parameter server architecture (old-style), not standard DDP
- D describes pipeline parallelism

---

## Q2 — Answer: C

**Answer:** C — Optimizer states and gradients.

**Why:** ZeRO progression: Stage 1 shards optimizer states; Stage 2 adds gradient sharding; Stage 3 adds parameter sharding. ZeRO-2 shards the two largest memory consumers (optimizer states and gradients) while keeping full parameters on each GPU for fast local computation.

**Why others are wrong:**
- A: Parameter-only sharding is not a ZeRO stage; ZeRO-3 shards parameters in addition to the others
- B: Gradient-only sharding is not a defined ZeRO stage
- D: That is ZeRO-3

---

## Q3 — Answer: C

**Answer:** C — 1.75GB.

**Why:** Under ZeRO-3, parameters are sharded across all 8 GPUs. 14GB / 8 = 1.75GB per GPU. Each GPU only materializes a single layer's parameters at a time during forward/backward, immediately discarding them after use.

**Why others are wrong:**
- A (7GB): that would be dividing by 2, not 8
- B (3.5GB): that would be dividing by 4
- D (0.87GB): that would be dividing by 16 — more GPUs than available

---

## Q4 — Answer: B

**Answer:** B — Gradient accumulation, mixed precision, and portability to multi-GPU without code changes.

**Why:** On a single GPU, Accelerate's main practical benefits are: (1) handling the GradScaler for FP16 mixed precision automatically, (2) clean gradient accumulation API, and (3) the same code runs on 4 GPUs next month with only a config change. It is a portability and boilerplate-reduction tool.

**Why others are wrong:**
- A: Accelerate does not implement CPU-GPU model parallelism
- C: Accelerate has no adaptive learning rate selection
- D: Accelerate still uses standard gradient-based optimization

---

## Q5 — Answer: A

**Answer:** A — Mathematically equivalent to a larger batch size for transformer models.

**Why:** Transformers use Layer Normalization (not Batch Normalization). LayerNorm statistics are per-sequence, not per-batch, so they are unaffected by how you split the batch across gradient accumulation steps. Gradients from N accumulation steps are numerically identical to gradients from a single N×batch_size batch.

**Why others are wrong:**
- B: Gradient accumulation does not reduce memory; it adds memory because gradients accumulate without being freed
- C: It is the same speed per effective batch (more steps but smaller per step)
- D: Gradient accumulation works on a single GPU — it has nothing to do with distributed training

---

## Q6 — Short Answer

ZeRO-3 shards parameters across GPUs, requiring an all-gather operation before every forward pass to materialize each layer's full weight matrix. On hardware with slow GPU interconnects (e.g., PCIe instead of NVLink, or across nodes via Infiniband), this all-gather can take longer than the compute time for small layers. ZeRO-2 avoids this overhead because parameters are replicated on each GPU. The result: on slow interconnects, ZeRO-2 with a higher-memory-per-GPU budget is faster than ZeRO-3.

---

## Q7 — Short Answer

14B params (FP16): 14e9 × 2 = 28GB  
Gradients (FP32): 14e9 × 4 = 56GB  
Optimizer states (FP32, Adam): 14e9 × 8 = 112GB  
Total single GPU: 196GB  

Under ZeRO-2 (shards optimizer + gradients, 2 GPUs):  
- Params: 28GB (full copy on each GPU)  
- Gradients: 56 / 2 = 28GB  
- Optimizer: 112 / 2 = 56GB  
- Total: 28 + 28 + 56 = **112GB per GPU**  

This does NOT fit on an A100 80GB. You would need ZeRO-3, or more GPUs, or 8-bit Adam (which halves optimizer state memory).

---

## Q8 — Short Answer

1. **Accelerate config change:** Run `accelerate config` and set `distributed_type: MULTI_GPU`, `num_processes: 4`
2. **Script launch:** Change `python train.py` to `accelerate launch --num_processes 4 train.py`
3. **Ensure model wrapping is via `accelerator.prepare()`:** No code changes needed if you already use Accelerate's API — Accelerate will automatically wrap with DDP

The key point: if your script already uses `accelerator.prepare()` and `accelerator.backward()`, switching from 1 GPU to 4 GPUs requires only a config/launch change.

---

## Q9 — Scenario Model Answer

**1. Per-GPU memory under DDP (13B, 4 GPUs):**
- Params (FP16): 13e9 × 2 = 26GB
- Gradients (FP32): 13e9 × 4 = 52GB
- Optimizer states (FP32): 13e9 × 8 = 104GB
- Total: 182GB per GPU (each GPU holds full copy under DDP)

4 × A100 80GB = 320GB total, but per-GPU requirement is 182GB. Does not fit on any single A100.

**2. Recommended ZeRO stage:**
ZeRO-2 on 4 GPUs: 26 + (52/4) + (104/4) = 26 + 13 + 26 = **65GB** — fits on A100 80GB with room for activations.

**3. Minimal Accelerate config change for DeepSpeed ZeRO-2:**
```yaml
distributed_type: DEEPSPEED
deepspeed_config:
  zero_stage: 2
  bf16:
    enabled: true
num_processes: 4
```

**4. Is 15% slowdown expected?**
Yes. ZeRO-2 introduces reduce-scatter communication for gradients (instead of all-reduce in DDP). Even on fast NVLink, this communication overhead is real. 15% overhead on 4 GPUs is acceptable — the trade-off is fitting a model that otherwise would not run at all. If the slowdown were >30%, you would investigate your network bandwidth or consider gradient compression.
