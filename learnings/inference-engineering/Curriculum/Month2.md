# Month 2 Curriculum: GPUs, Mixed Precision, and Profiling

## Month objective

Month 2 changes the question from **“Is the model mathematically correct?”** to **“What does the machine actually do when the model runs, where does time go, and why?”** By the end, a training step should no longer look like one Python call: it should look like CPU input work, host-to-device transfer, CUDA kernel launches, execution on SMs, reads and writes through a memory hierarchy, synchronization, and optimizer work.

The final outcome is one controlled CIFAR-10 experiment comparing FP32 with mixed precision, supported by benchmark measurements and profiler traces—not merely an accuracy notebook. The goal is not to become a CUDA kernel expert this month; it is to acquire the hardware and measurement model needed for every later inference-engineering phase.

## Starting position

The available learning log shows that the PyTorch basics, CIFAR-10 training loop, model save/load, basic multi-GPU concepts, and einops basics have been completed. It also shows that Month 1's five-case, 20-seed autograd parity harness was still unfinished at the last entry.[^1]

Complete the following gate before Session 3 of this plan:

- Finish all five Month 1 parity cases: linear, two-layer MLP, ReLU/softmax MLP, manual L2 regularization, and toy RNN.
- Run all cases across 20 seeds with `max_abs_diff < 1e-6`.
- Explain what `.backward()` accumulates into `.grad`, why gradients must be cleared, and where `.detach()` cuts the graph.

The CIFAR-10 loop already written in Month 1 should become the starting point for Project A2. Do not copy a polished trainer: refactor the existing loop so that each new system feature is understood as a change to known code.

## Learning outcomes

At the end of Month 2, you should be able to:

- Draw the path from a PyTorch operation to an ATen operation, CUDA launch, kernel, SM, warp, and memory access.
- Explain grids, thread blocks, warps, SMs, registers, shared memory/SRAM, cache, and global device memory.
- Estimate the FLOPs and transferred bytes of a simple operation and use arithmetic intensity to predict whether compute or memory is the likely limiter.
- Distinguish GPU utilization, SM occupancy, memory-bandwidth utilization, and actual useful throughput.
- Explain FP32, FP16, BF16, autocasting, accumulation precision, underflow, overflow, and gradient scaling.
- Add AMP correctly without manually converting the model or input tensors to half precision.
- Benchmark CUDA code without accidentally timing only asynchronous kernel dispatch.
- Capture and interpret a PyTorch Profiler trace with CPU and CUDA activity, input shapes, memory, and named regions.
- Separate data loading, host-to-device transfer, forward/loss, backward, and optimizer time.
- Explain from first principles why halving an operand's bytes cannot guarantee a 2x end-to-end speedup.

## Important hardware correction

The source curriculum says “T4 or better” and specifically asks for BF16. These requirements do not fully agree. NVIDIA's T4 specification advertises FP32, FP16, INT8, and INT4 execution, while native CUDA BF16 types require compute capability 8.0 or newer. Therefore:[^2][^3]

- **Strict BF16 project:** use an Ampere-or-newer GPU on which `torch.cuda.is_bf16_supported(including_emulation=False)` returns `True`; PyTorch provides this API specifically to test BF16 support.[^4]
- **Free T4 fallback:** run FP32 versus FP16 autocast with `GradScaler`, clearly label the deviation, and repeat only the final benchmark/profile on a native-BF16 machine if one becomes available.
- **Do not silently call a T4 FP16 run “BF16.”** Hardware identity and dtype are experimental metadata.

Run this preflight before doing any measurements:

```python
import platform
import torch

print("Python:", platform.python_version())
print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Capability:", torch.cuda.get_device_capability(0))
    print("BF16 native:", torch.cuda.is_bf16_supported(including_emulation=False))
    print("cuDNN:", torch.backends.cudnn.version())
```

Save the output in the final report. Never compare runs performed on different GPU types as if precision were the only changed variable.

## Work structure

Follow the sessions in order. A session is 60–120 focused minutes; training jobs can continue after the study portion. The plan is approximately 15 hours per week for four weeks, but session numbering makes it usable even if office work causes an uneven calendar.

| Stage | Sessions | Main result |
|---|---:|---|
| Closure and setup | 0–2 | Month 1 closed; reproducible GPU environment |
| GPU tooling | 3–6 | PyTorch-to-kernel execution map |
| GPU architecture | 7–11 | Compute/memory and roofline mental model |
| Mixed precision | 12–16 | Correct FP32 and AMP training paths |
| Profiling | 17–20 | Labeled traces and bottleneck diagnosis |
| Project A2 | 21–26 | Controlled experiment, report, oral defense |

## Stage 0: Close Month 1

### Session 0 — Parity harness

- Finish the five-case autograd parity harness.
- Run `pytest -q` and save the terminal output.
- For every parameter, verify both its shape and numerical gradient—not only loss equality.
- Write one sentence per test stating the local derivative and the upstream gradient it receives.

**Exit check:** Why is matching forward output insufficient evidence that the backward pass is correct?

### Session 1 — Refactor the existing trainer

Refactor the Month 1 CIFAR-10 notebook into these functions without changing behavior:

```text
set_seed
build_dataloaders
build_model
train_one_epoch
evaluate
save_checkpoint
load_checkpoint
```

Add a `config` object containing seed, batch size, epochs, learning rate, precision mode, workers, device, and output directory. The same training function must eventually support both FP32 and AMP.

**Assignment M2.0:** Prove checkpoint correctness by training briefly, saving, loading into a new model instance, and confirming identical evaluation logits for one fixed batch.

### Session 2 — Reproducible environment

- Choose one stable GPU runtime for the final paired experiment.
- Save the hardware preflight output.
- Record package versions with `pip freeze` or `uv pip freeze`.
- Fix random seeds for Python, NumPy, PyTorch CPU, and CUDA.
- Create output folders for checkpoints, benchmark CSV, traces, profiler tables, and the final report.

Suggested structure:

```text
month2/
  src/
    data.py
    model.py
    train.py
    benchmark.py
    profile_run.py
  tests/
  outputs/
    fp32/
    amp/
    traces/
  report.md
  requirements.txt
  README.md
```

## Stage 1: GPU tooling

GPU MODE Lecture 1 is explicitly the lecture on profiling and integrating CUDA kernels into PyTorch, with its notebook/slides located in `lecture_001`. Its examples move among ordinary PyTorch, Triton, inline C++/CUDA extensions, and NVIDIA profiling tools; `load_inline` handles source generation, Python binding, compilation, and loading of an extension.[^5][^6]

### Session 3 — Watch Lecture 1 once

Watch without coding. Capture answers to only these questions:

1. What layers exist between `y = x * x` in Python and instructions executing on a GPU?
2. What question does PyTorch Profiler answer?
3. What question does Nsight Systems answer?
4. What question does Nsight Compute answer?
5. Why might someone try Triton before handwritten CUDA?
6. What costs exist even for an almost-empty CUDA kernel?

Do not pause to understand every line of CUDA. The first pass is a map of the territory.

### Session 4 — Re-run the square examples

Work through the Lecture 1 repository examples in this order:

1. Plain PyTorch square.
2. PyTorch Profiler square.
3. Triton square.
4. `torch.utils.cpp_extension.load_inline` square.
5. Nsight Systems/Compute command examples if the runtime permits them.

For each implementation, record:

- Input shape and dtype.
- Correctness check against `x.square()`.
- First-run time versus warmed-up time.
- Number/names of visible kernels.
- What compilation or launch overhead occurred.

**Assignment M2.1:** Make the square implementation fail in three ways—CPU tensor passed to CUDA extension, non-contiguous tensor, and unsupported dtype. For each failure, write where validation should happen and whether the error belongs to Python, the C++ binding, or the CUDA kernel boundary.

### Session 5 — Tool hierarchy

Create a one-page “profiling zoom lens”:

| Tool | Scope | Primary question |
|---|---|---|
| Python timer | End-to-end | How long did the caller wait? |
| CUDA events | Device interval | How long did queued GPU work take? |
| PyTorch Profiler | Framework operators + kernels | Which model regions/operators consume time and memory? |
| Nsight Systems | Whole-system timeline | Where are CPU, CUDA API, copies, kernels, and idle gaps? |
| Nsight Compute | One/few kernels | Why is this kernel slow at the hardware-counter level? |

The PyTorch Profiler records CPU and CUDA activities, can attach input shapes and memory information, and supports custom labels through `record_function`. Nsight Compute is a later zoom level; Month 2 requires familiarity with its purpose, not mastery.[^7]

### Session 6 — Asynchronous execution lab

Time one CUDA operation incorrectly with `time.perf_counter()`, then time it correctly after warm-up using either CUDA events or explicit synchronization. CUDA work is asynchronous with respect to the host, so an unsynchronized CPU timer may measure enqueue time rather than execution; NVIDIA recommends CUDA events and warm-up for device timing.[^8]

**Deliverable:** a table comparing naive CPU timing, synchronized wall time, and CUDA-event time. Explain why they differ.

**First-principles check:** If Python immediately returns after launching a kernel, which processor is doing work, and what event forces the CPU to wait?

## Stage 2: GPU architecture

CUDA's execution model treats the GPU as a collection of streaming multiprocessors. A thread block executes on one SM, threads in a block are grouped into warps of 32, and each SM has local registers and shared memory. These are the minimum facts needed to reason about Lecture 4 rather than memorizing terminology.[^9]

### Session 7 — Pre-read before Lecture 4

Learn this hierarchy and redraw it from memory:

```text
CPU program
  -> launches kernel
GPU grid
  -> many thread blocks
thread block
  -> assigned to one SM
  -> divided into warps
warp
  -> 32 lanes/threads executing SIMT instructions
thread
  -> registers and per-thread state
```

Answer:

- Why can threads in one block cooperate cheaply?
- Why can two arbitrary blocks not assume simultaneous execution?
- Why should common block sizes generally be multiples of 32?
- What happens when lanes in one warp take different branches?

### Session 8 — Watch Lecture 4

Watch the lecture without implementing. Focus on the causal chain:

```text
operation shape
  -> amount of work and data movement
  -> kernel launch/grid
  -> blocks scheduled on SMs
  -> warps execute
  -> registers/shared memory/global memory are accessed
  -> one resource becomes the throughput limiter
```

Capture the lecture's examples around approximate GELU fusion, launch latency, tiled matrix multiplication, and occupancy. The notebook asks why an unfused GELU is slow, measures empty-kernel launch latency, and explores tiled matmul and resource-limited occupancy.[^10]

### Session 9 — Memory hierarchy worksheet

Build a conceptual comparison; do not memorize architecture-specific nanosecond values.

| Tier | Scope | Relative capacity | Relative speed | Main use |
|---|---|---|---|---|
| Registers | One thread | Tiny | Fastest | Scalars and local intermediates |
| Shared memory/SRAM | One block on an SM | Small | Very fast | Explicit data reuse and cooperation |
| L1/L2 cache | SM/device | Medium | Intermediate | Reuse captured by hardware caching |
| Global device memory | Whole GPU | Large | Slowest on-device tier | Parameters, activations, inputs, outputs |
| Host memory | CPU side | Very large | Across PCIe/interconnect | Dataset and application state |

For each of these operations, estimate input/output bytes and useful arithmetic operations:

- `y = x + 1`
- `y = gelu(x)` written as several separate tensor operations
- Fused `y = gelu(x)`
- Matrix multiplication `C = A @ B`
- 2D convolution

The purpose is to discover reuse. Fusion can reduce intermediate writes/reads and launch overhead; tiling can load a value from global memory once and reuse it many times from shared memory.

### Session 10 — Roofline reasoning

Learn these definitions:

\[
\text{Arithmetic intensity} = \frac{\text{useful FLOPs}}{\text{bytes moved from limiting memory tier}}
\]

\[
\text{Attainable performance} \leq \min(\text{peak compute},\; \text{memory bandwidth} \times \text{arithmetic intensity})
\]

Use them qualitatively:

- Low arithmetic intensity suggests memory bandwidth can dominate.
- High arithmetic intensity provides a chance to become compute-bound.
- Small workloads can instead be launch- or CPU-bound because they do not expose enough parallel work.
- A high GPU-Util number alone does not prove good efficiency.

`nvidia-smi` defines GPU utilization as the percentage of a sampling period in which one or more kernels executed; it does not say how many SMs were productively used. A single continuously running but inefficient kernel can therefore report high utilization.[^11]

**Assignment M2.2:** Predict the likely bottleneck for vector add, large matmul, tiny matmul, unfused elementwise chain, and DataLoader-starved CNN. Then design one measurement that could falsify each prediction.

### Session 11 — Lecture 4 coding pass

Return to the Lecture 4 notebook and run the GELU, empty-kernel, and tiled-matmul sections. For each:

- Verify correctness before measuring speed.
- Warm up before collecting timing.
- Change problem size or tile size.
- Explain the result using launch overhead, data reuse, occupancy, or bandwidth.

**Exit check:** Why can a fused kernel be faster even when it performs approximately the same mathematical operations?

## Stage 3: Mixed precision

PyTorch AMP assigns operation-specific dtypes inside an autocast region: operations such as convolutions and linear layers can use FP16/BF16, while numerically sensitive reductions may remain FP32. Autocast should cover the forward pass and loss, not the optimizer or a manually half-cast model; backward is run outside the autocast context.[^12][^13]

### Session 12 — Floating-point foundations

For FP32, FP16, and BF16, learn:

- Sign, exponent, and fraction/significand roles.
- Dynamic range versus precision.
- Overflow: magnitude is too large to represent.
- Underflow: a small nonzero value rounds to zero or a subnormal.
- Why BF16 retains a range similar to FP32 but has fewer fraction bits.
- Why FP16 has more fraction precision than BF16 but much less exponent range.
- Why accumulation may use FP32 even when multiplication operands use a lower precision.

**Assignment M2.3a:** Without code, predict which is more at risk for gradient underflow—FP16 or BF16—and which provides more fine-grained precision near 1. Then verify with `torch.finfo` and small tensor experiments.

### Session 13 — AMP recipe exactly as written

Run PyTorch's official AMP recipe in FP32 and mixed precision. It demonstrates synchronized timing, peak allocated memory, autocast, and `GradScaler`; the recipe notes that small or CPU-bound models may show little benefit because mixed precision helps most when GPU work is large enough to saturate the hardware.[^12]

Record:

- Runtime.
- Peak allocated tensor memory.
- Output dtype of a linear/conv operation.
- Loss dtype.
- Whether a scaler is enabled.
- GPU model and participating dimensions.

Then deliberately break it:

- Put `.backward()` inside autocast.
- Manually call `.half()` or `.bfloat16()` on the full model.
- Use FP16 without a scaler and construct values small enough to underflow.
- Read/clip scaled gradients before `scaler.unscale_(optimizer)`.
- Call `.item()` every training step and observe synchronization overhead.

### Session 14 — Autocast mental model

Implement one training step with a precision switch:

```python
amp_dtype = (
    torch.bfloat16
    if torch.cuda.is_bf16_supported(including_emulation=False)
    else torch.float16
)
use_scaler = use_amp and amp_dtype == torch.float16
scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

optimizer.zero_grad(set_to_none=True)
with torch.autocast(
    device_type="cuda",
    dtype=amp_dtype,
    enabled=use_amp,
):
    logits = model(images)
    loss = criterion(logits, targets)

if use_scaler:
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
else:
    loss.backward()
    optimizer.step()
```

For FP16, gradient scaling raises small gradient magnitudes before backward, unscales before the update, and can skip an update when infinities or NaNs are detected. BF16 ordinarily does not need loss scaling because its exponent range is far larger than FP16's; still monitor non-finite losses and gradients.[^14][^15]

**First-principles check:** Gradient scaling changes the numbers used during backward. Why does it not intentionally change the effective parameter update?

### Session 15 — Precision microbenchmarks

For a representative convolution and matrix multiplication, sweep:

- FP32, FP16 autocast, and BF16 autocast when supported.
- Batch sizes such as 32, 64, 128, and the largest safe size.
- At least one Tensor-Core-unfriendly versus friendlier dimension where relevant.

Measure median device time after warm-up and peak allocated memory. Do not draw conclusions from one iteration. The T4, for example, has hardware acceleration for mixed FP16/FP32 work; BF16 conclusions require a native-BF16 GPU.[^2]

### Session 16 — Explain non-2x speedup

Write the explanation before running Project A2:

- Lower precision does not halve every byte: labels, some activations/operations, master parameters, gradients, and optimizer state may remain in other formats.
- Autocast deliberately keeps selected operations in FP32 for numerical behavior.[^12]
- Data loading, Python, CPU transforms, host-to-device copies, kernel launch overhead, loss bookkeeping, and parts of the optimizer do not automatically become twice as fast.
- A small CIFAR-10 CNN may not create enough work to saturate Tensor Cores; PyTorch's AMP recipe explicitly warns that CPU-bound or small workloads can see minor speedup.[^12]
- If the baseline is memory-bound, fewer bytes may help; if it is launch-, input-, or non-Tensor-Core-op-bound, the same reduction may have little effect.
- End-to-end speedup obeys the fraction of time actually accelerated, not the peak throughput ratio of one hardware unit.

## Stage 4: PyTorch profiling

The official profiler recipe covers operator timing, memory, Chrome/Perfetto trace export, stack traces, and scheduled profiling for long-running jobs. A schedule is essential here: profiling an entire 15-epoch training run creates unnecessary overhead and a huge trace.[^7]

### Session 17 — Profiler recipe

Run the official recipe top to bottom. Be able to distinguish:

- CPU time versus CUDA time.
- Self time versus total time.
- Operator (`aten::conv2d`) versus device kernel.
- Number of calls versus average time.
- Allocated memory versus reserved memory.
- Input-shape grouping.
- Framework stack trace versus execution timeline.

The profiler's `record_shapes=True` allows grouping the same operator by input shape, and `profile_memory=True` tracks tensor allocation/deallocation. Stack capture is useful but adds overhead, so enable it only in a diagnostic trace.[^7]

### Session 18 — Instrument the trainer

Create named regions around the actual stages:

```python
with record_function("data_wait"):
    images, targets = next(data_iter)

with record_function("host_to_device"):
    images = images.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)

with record_function("zero_grad"):
    optimizer.zero_grad(set_to_none=True)

with record_function("forward_and_loss"):
    with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
        logits = model(images)
        loss = criterion(logits, targets)

with record_function("backward"):
    backward_step(loss)

with record_function("optimizer_step"):
    optimizer_step()
```

Fetching must be explicitly timed with an iterator if you want `data_wait` to appear as a region. In an ordinary `for images, targets in loader` loop, the fetch happens before control enters the loop body.

### Session 19 — Scheduled trace

Capture only a small steady-state window:

```python
activities = [
    torch.profiler.ProfilerActivity.CPU,
    torch.profiler.ProfilerActivity.CUDA,
]

with torch.profiler.profile(
    activities=activities,
    schedule=torch.profiler.schedule(wait=2, warmup=2, active=5, repeat=1),
    on_trace_ready=lambda p: p.export_chrome_trace(
        f"outputs/traces/{precision}_step{p.step_num}.json"
    ),
    record_shapes=True,
    profile_memory=True,
    with_stack=False,
) as prof:
    for step in range(9):
        run_one_training_step(...)
        prof.step()
```

PyTorch's schedule separates wait, warm-up, and active steps so initial profiling overhead does not dominate the captured interval. Open the resulting JSON in Perfetto or `chrome://tracing`; current PyTorch documentation recommends Perfetto/Chrome traces and marks the older TensorBoard profiler integration as deprecated.[^16][^7]

### Session 20 — Read the trace

For both precision modes, answer:

- Where does one training step begin and end?
- Is there an idle gap before forward?
- Are host-to-device copies visible?
- Which CUDA kernels dominate forward?
- Which kernels dominate backward?
- Does the optimizer create many small kernels?
- Which operator has the highest self CUDA time?
- Which input shape corresponds to that operator?
- Are kernels separated by CPU launch gaps?
- Did mixed precision change kernel names, durations, or memory traffic?

**Assignment M2.4:** Write a five-sentence diagnosis before trying any optimization. Every sentence must connect an observation in the trace to a hypothesis; avoid generic advice such as “increase utilization.”

## Project A2: Controlled trainer

### Step 21 — Build the model

Use a small CNN with exactly four convolution layers and no pretrained weights. A suitable starting structure is:

```text
Input: 3 x 32 x 32
Conv 3->64, 3x3, padding=1 -> BatchNorm -> ReLU
Conv 64->64, 3x3, padding=1 -> BatchNorm -> ReLU -> MaxPool
Conv 64->128, 3x3, padding=1 -> BatchNorm -> ReLU
Conv 128->128, 3x3, padding=1 -> BatchNorm -> ReLU -> MaxPool
Adaptive average pool -> Linear 128->10
```

Before training, print each stage's tensor shape and calculate the parameter count. Explain why convolution reuses each filter across spatial locations and why adaptive average pooling reduces dependence on a manually calculated flatten size.

### Step 22 — Establish accuracy

Use training augmentation, normalization, and a held-out test transform without random augmentation. Start with one conventional optimizer/scheduler pair—for example, SGD with momentum and weight decay plus cosine decay, or AdamW plus cosine decay—and tune only until FP32 reaches at least 70% test accuracy within 15 epochs.

Rules:

- Test data never influences gradient updates.
- Do not add pretrained weights.
- Do not increase beyond 15 epochs.
- Record every hyperparameter.
- Once the FP32 configuration crosses the bar, freeze it for the precision comparison.

### Step 23 — Freeze the experiment

Create one initial model state and use it for both trials. Keep constant:

- GPU and software environment.
- Initial model weights.
- Dataset split and transformations.
- Seed and batch order as far as practical.
- Batch size.
- Optimizer, scheduler, learning rate, and epoch count.
- Logging frequency.

Change only the precision path. Run the clean training jobs **without the profiler enabled**; profiler overhead makes profiled runtime unsuitable as the headline speed comparison.

### Step 24 — Benchmark cleanly

For each precision mode, collect:

- Test accuracy after every epoch.
- Mean epoch time.
- Median steady-state training-step time after warm-up.
- Images per second.
- Peak `torch.cuda.max_memory_allocated()`.
- Mean and median GPU-Util samples from the same collection method.
- GPU name, dtype, batch size, and package versions.

Use warm-up and CUDA synchronization/events for timing. The official AMP recipe synchronizes before and after a timed region and resets peak allocated memory before measuring.[^12]

If collecting `nvidia-smi` utilization, state its precise meaning: percentage of sampled time with at least one kernel active—not percentage of theoretical GPU FLOPs achieved. Do not use it alone to claim efficiency.[^11]

Compute:

\[
\text{speedup} = \frac{\text{median FP32 step time}}{\text{median AMP step time}}
\]

\[
\text{memory reduction} = 1 - \frac{\text{AMP peak allocated memory}}{\text{FP32 peak allocated memory}}
\]

### Step 25 — Capture paired traces

Capture one scheduled FP32 trace and one scheduled AMP trace with the same batch size and profiler settings. Export:

- `fp32_trace.json`
- `amp_trace.json`
- `fp32_top_ops.txt`
- `amp_top_ops.txt`

Each table should include the top operators by self CUDA time and, separately, by CPU time. The trace must visibly contain the custom data, transfer, forward, backward, and optimizer regions.

### Step 26 — Write and defend

Use this report structure:

```text
1. Hardware/software environment
2. Model and parameter count
3. Data pipeline and augmentation
4. Shared training configuration
5. FP32 results
6. AMP results
7. Benchmark method
8. Profiler method
9. Trace observations
10. Bottleneck diagnosis
11. Why speedup is not 2x
12. Limitations and next experiment
```

Required result table:

| Metric | FP32 | AMP | Interpretation |
|---|---:|---:|---|
| Final test accuracy | Measured | Measured | Accuracy delta |
| Median step time | Measured | Measured | Headline latency |
| Images/sec | Measured | Measured | Throughput |
| Speedup | 1.00x | Measured | FP32 time / AMP time |
| Peak allocated memory | Measured | Measured | Tensor-memory effect |
| Mean GPU-Util | Measured | Measured | Busy-time signal only |
| Data wait share | Measured | Measured | Input bottleneck evidence |
| Forward share | Measured | Measured | Precision-sensitive work |
| Backward share | Measured | Measured | Precision-sensitive work |
| Optimizer share | Measured | Measured | Often less accelerated |

Do not invent expected numbers. The point is to explain the measurements obtained on the actual hardware.

## Acceptance checklist

Project A2 is complete only when every required box is checked:

- [ ] Month 1 parity harness passes all five cases across 20 seeds.
- [ ] Four-convolution CNN uses no pretrained weights.
- [ ] Test accuracy is at least 70% within 15 epochs.
- [ ] FP32 and AMP trials use the same frozen configuration and initial weights.
- [ ] The actual AMP dtype is recorded.
- [ ] A strict BF16 run uses a GPU reporting native BF16 support; otherwise the FP16 fallback is clearly declared.
- [ ] Timing excludes initial warm-up and correctly handles asynchronous CUDA execution.
- [ ] Clean benchmark timing is separated from profiler timing.
- [ ] Peak allocated memory is reported for both modes.
- [ ] GPU utilization is reported with its measurement method and definition.
- [ ] One FP32 and one AMP Chrome/Perfetto trace open successfully.
- [ ] Both traces contain labeled data, transfer, forward, backward, and optimizer regions.
- [ ] A top-operator table is saved for each mode.
- [ ] The measured speedup ratio is stated.
- [ ] The non-2x result is explained through bottleneck fractions, unchanged work, mixed operation dtypes, workload size, and hardware support.
- [ ] The trace can be narrated without notes.

## Glossary

### Execution model

| Term | Meaning |
|---|---|
| Host | CPU and its memory; launches GPU work. |
| Device | GPU and its memory. |
| Kernel | Function executed in parallel by GPU threads. |
| Grid | All thread blocks launched for one kernel. |
| Thread block | Group of threads assigned to one SM; its threads can synchronize and share on-chip shared memory. |
| Thread | One logical execution instance of a kernel. |
| Warp | Hardware scheduling group of 32 CUDA threads on NVIDIA GPUs[^9]. |
| SIMT | Single-instruction, multiple-thread execution model; warp lanes run a common instruction stream but retain per-thread state. |
| SM | Streaming multiprocessor; schedules warps and contains execution units, registers, and shared memory[^9]. |
| Occupancy | Ratio of active resident warps to the architecture's maximum possible resident warps; useful for latency hiding but not identical to performance. |
| Warp divergence | Lanes in a warp take different control-flow branches, forcing paths to be handled separately. |
| Kernel launch overhead | CPU/runtime cost and GPU scheduling latency before useful kernel work; important for tiny operations. |

### Memory and performance

| Term | Meaning |
|---|---|
| Register | Fast, per-thread on-chip storage. |
| SRAM/shared memory | Fast on-chip memory shared by threads in a block and explicitly managed by the kernel. |
| Cache | Hardware-managed storage that retains recently/repeatedly accessed data. |
| Global/device memory | Large off-chip GPU memory holding tensors; high bandwidth but much slower than registers/shared memory. |
| HBM | High-bandwidth memory used by many data-center GPUs; a form of global device memory. Do not call every GPU's global memory HBM—a T4, for example, uses GDDR memory. |
| Memory bandwidth | Bytes transferable per unit time between a memory tier and compute units. |
| Latency | Delay between requesting an operation/data and receiving its result. |
| Throughput | Work completed per unit time, such as images/sec. |
| FLOP | One floating-point arithmetic operation under the counting convention being used. |
| FLOPS | Floating-point operations per second. |
| Arithmetic intensity | Useful FLOPs divided by bytes moved through a selected memory tier. |
| Compute-bound | Performance limited mainly by execution throughput. |
| Memory-bound | Performance limited mainly by data movement bandwidth. |
| Launch-bound | Work is too small, so launch/dispatch overhead dominates. |
| CPU/input-bound | GPU waits because host code, transforms, storage, or transfer cannot supply work fast enough. |
| Fusion | Combining multiple operations into fewer kernels to reduce launches and intermediate memory traffic. |
| Tiling | Dividing data into reusable blocks that fit in faster memory. |
| Coalescing | Organizing warp memory accesses so they can be served with few efficient memory transactions. |
| Roofline model | Bound relating peak compute, memory bandwidth, and arithmetic intensity. |
| GPU utilization | Busy-time metric: fraction of sampled time during which at least one kernel ran[^11]. |
| SM efficiency/activity | Finer metric describing how extensively SMs are active; not the same as GPU utilization. |

### Precision

| Term | Meaning |
|---|---|
| FP32 | 32-bit floating-point format commonly used as the default training precision. |
| FP16 | 16-bit IEEE half precision; smaller range than BF16 and therefore more vulnerable to gradient underflow. |
| BF16 | 16-bit format with an FP32-like exponent range but fewer fraction bits; native CUDA BF16 requires newer hardware[^3]. |
| Mixed precision | Use of multiple floating formats in one execution according to speed and numerical needs. |
| Tensor Core | Specialized hardware for matrix-like mixed-precision operations. |
| Autocast | PyTorch context that selects an operation-specific lower or full precision automatically[^13]. |
| Accumulation precision | Format used to add intermediate products, which may be wider than operand precision. |
| Dynamic range | Span between smallest and largest representable magnitudes. |
| Precision | Fineness with which nearby values can be distinguished. |
| Overflow | Result magnitude exceeds representable range, often producing infinity. |
| Underflow | Tiny nonzero result becomes subnormal or zero. |
| Gradient scaling | Multiplying loss before backward to prevent FP16 gradients from underflowing, then unscaling before the update[^15]. |
| NaN/Inf | Non-finite values signalling invalid arithmetic or overflow. |

### Profiling

| Term | Meaning |
|---|---|
| Benchmark | Controlled measurement designed to compare performance. |
| Profile | Detailed observation intended to locate where and why time or memory is consumed. |
| Warm-up | Unmeasured iterations used to absorb initialization, compilation, cache, and clock-stabilization effects. |
| Synchronization | Waiting until previously queued GPU work completes. |
| CUDA event | Device-timestamped marker used for GPU timing. |
| Trace | Time-ordered record of CPU operations, CUDA calls, copies, kernels, and custom ranges. |
| `ProfilerActivity.CPU/CUDA` | Activities telling PyTorch Profiler which host/device events to record[^7]. |
| `record_function` | Context manager that adds a custom named range to a trace[^7]. |
| `record_shapes` | Profiler option recording operator input shapes[^7]. |
| `profile_memory` | Profiler option tracking tensor allocation and release[^7]. |
| Self time | Time directly spent in an event, excluding child events. |
| Total time | Event time including nested/child operations. |
| Profiler schedule | Wait/warm-up/active cycle used to sample long-running jobs without tracing everything[^7]. |
| Nsight Systems | System-level timeline profiler for CPU, CUDA APIs, transfers, and kernels. |
| Nsight Compute | Kernel-level profiler exposing hardware counters and bottleneck evidence. |
| Perfetto | Browser trace viewer suitable for PyTorch-exported Chrome JSON traces[^16]. |

## Learning materials

Use the materials in this exact order:

1. **GPU MODE Lecture 1 — Profiling and Integrating CUDA Kernels in PyTorch:** watch the lecture; clone `gpu-mode/lectures`; inspect `lecture_001`; run the PyTorch, Triton, `load_inline`, and profiler examples.[^6][^5]
2. **GPU MODE Lecture 4 — Intro to Compute and Memory Architecture:** watch once for structure, complete the execution/memory worksheet, then return for the GELU fusion, launch latency, tiled matmul, and occupancy code.[^10]
3. **NVIDIA CUDA Programming Guide, Programming Model:** read only the sections on SMs, thread blocks, warps, and memory scope as a reference when Lecture 4 is unclear.[^9]
4. **PyTorch Automatic Mixed Precision recipe:** run every section, including timing, peak memory, autocast, scaler, gradient inspection, checkpointing, and troubleshooting.[^12]
5. **PyTorch AMP reference/examples:** use this to verify current APIs and which operations or regions should run under autocast.[^13][^14]
6. **PyTorch Profiler recipe:** complete timing, memory, trace export, stack traces, and scheduled profiling.[^7]
7. **Perfetto or Chrome trace viewer:** use it to narrate the two final JSON traces; TensorBoard profiler integration is now deprecated in PyTorch's documentation.[^16]

## Oral examination

Before moving to Month 3, answer these aloud without notes:

1. Why is a GPU not simply “a faster CPU”?
2. Starting from `loss.backward()`, what CPU and GPU events occur before gradients are ready?
3. Why do GPUs schedule threads in warps, and what resource owns a thread block?
4. Why is shared memory useful if global memory already exists?
5. How can tiling increase arithmetic intensity?
6. How can an operation show 100% GPU utilization and still be inefficient?
7. Why can an unfused elementwise expression be memory- and launch-bound?
8. Why must CUDA timing account for asynchronous execution?
9. What exactly does autocast decide, and what does it not decide?
10. Why does FP16 often need gradient scaling?
11. Why is BF16 not simply “a more accurate FP16”?
12. Why should backward occur outside the autocast context?
13. Why can AMP reduce peak activation memory without halving total training memory?
14. Why is profiler timing not the same as clean benchmark timing?
15. Looking at a trace, how would you distinguish input starvation from a slow convolution?
16. Why is the measured FP32-to-AMP speedup not guaranteed to be 2x?

Month 2 is complete when the answers connect measurements to mechanisms. Merely reaching 70% accuracy or producing a JSON trace is insufficient; the acceptance bar is the ability to explain why the trace looks the way it does.

---

## References

1. [daily logs.md](daily logs.md)

2. [t4-tensor-core-datasheet-951643.pdf](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf)

3. [5.4. C/C++ Language Extensions — CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html)

4. [torch.cuda.is_bf16_supported — PyTorch 2.14 documentation](https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_bf16_supported.html) - Return a bool indicating if the current CUDA/ROCm device supports dtype bfloat16. Rate this Page.. S...

5. [README.md - gpu-mode/lectures - GitHub](https://github.com/gpu-mode/lectures/blob/main/README.md) - Lecture 1: Profiling and Integrating CUDA kernels in PyTorch. Speaker: Mark Saroufim; Notebook and s...

6. [GPU MODE Lecture 1: How to profile CUDA kernels in PyTorch](https://christianjmills.com/posts/cuda-mode-notes/lecture-001/) - Lecture #1 provides a practical introduction to integrating and profiling custom CUDA kernels within...

7. [PyTorch Profiler — PyTorch Tutorials 2.14.0+cu130 documentation](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) - This recipe explains how to use PyTorch profiler and measure the time and memory consumption of the ...

8. [Commonly Used Command-Line...](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/benchmarking.html)

9. [1.2. Programming Model — CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)

10. [lectures/lecture_004/cuda-mode-session-4.ipynb at main · gpu-mode/lectures](https://github.com/gpu-mode/lectures/blob/main/lecture_004/cuda-mode-session-4.ipynb) - Material for gpu-mode lectures. Contribute to gpu-mode/lectures development by creating an account o...

11. [NVIDIA Documentation Hub](https://docs.nvidia.com/deploy/nvidia-smi/)

12. [amp_recipe.py](https://docs.pytorch.org/tutorials/_downloads/cadb3a57e7a6d7c149b5ae377caf36a8/amp_recipe.py)

13. [Automatic Mixed Precision package - torch.amp — PyTorch 2.8 ...](https://docs.pytorch.org/docs/stable/amp.html?highlight=gradscalertorch.cuda.amp.GradScaler)

14. [Automatic Mixed Precision examples — PyTorch 2.14 documentation](https://docs.pytorch.org/docs/stable/notes/amp_examples.html) - Gradient scaling improves convergence for networks with float16 (by default on CUDA and XPU) gradien...

15. [amp.md.txt](https://docs.pytorch.org/docs/2.9/_sources/amp.md.txt)

16. [PyTorch Profiler With TensorBoard](https://docs.pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html?highlight=profile)

