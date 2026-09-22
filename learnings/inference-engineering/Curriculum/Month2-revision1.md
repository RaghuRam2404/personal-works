# Month 2 — GPUs, Mixed Precision, and Profiling

## Goal

Month 2 changes the question from **“Is the model mathematically correct?”** to **“What is the computer doing when the model runs, where does the time go, and why?”**

The final outcome is a controlled CIFAR-10 experiment comparing FP32 with mixed-precision training. You will collect profiler traces, compare performance, and explain why mixed precision does not automatically produce a 2x end-to-end speedup.

Month 2 is **not** a CUDA-kernel mastery month. Triton programming, custom CUDA extensions, tiled matrix multiplication, roofline calculations, occupancy tuning, and detailed Nsight Compute analysis are deferred to the later GPU-programming phase.

## Required outcome

By the end of the month, you must be able to:

- Explain the basic path from a PyTorch operation to GPU kernel execution.
- Explain kernels, SMs, warps, registers, shared memory, and global GPU memory.
- Distinguish basic compute-, memory-, launch-, and input-bound behavior.
- Explain FP32, FP16, BF16, autocast, underflow, overflow, and gradient scaling.
- Add AMP correctly to a PyTorch training loop.
- Measure CUDA execution without relying on an unsynchronized Python timer.
- Capture and open a PyTorch Profiler trace with `record_shapes=True`.
- Identify data loading, transfer, forward, backward, and optimizer work in a trace.
- Compare FP32 with mixed precision on CIFAR-10.
- Explain why the measured speedup is not 2x.

## Work structure

Sessions 0–3 are retained from the original plan. Sessions 4–10 replace the previous over-scoped Sessions 4–26.

| Stage | Sessions | Result |
|---|---:|---|
| Month 1 closure and setup | 0–2 | Correctness harness and reproducible trainer |
| GPU tooling introduction | 3–4 | Basic execution and profiling-tool mental model |
| GPU architecture | 5 | Basic hardware and memory model |
| Mixed precision | 6 | AMP integrated into the trainer |
| PyTorch Profiler | 7 | One working trace and an instrumented trainer |
| Project A2 | 8–10 | FP32/AMP results, paired traces, and report |

At your current pace, Sessions 4–10 should require approximately 21–28 focused hours. Model training can run unattended and does not count as focused study time.

---

## Stage 0 — Close Month 1

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

## Stage 1 — GPU tooling

GPU MODE Lecture 1 introduces several routes from PyTorch to GPU execution and several profiling tools. The goal this month is to understand what each route or tool is for—not to master all of them.

### Session 3 — Watch Lecture 1 once

Watch without coding. Capture answers to only these questions:

1. What layers exist between `y = x * x` in Python and instructions executing on a GPU?
2. What question does PyTorch Profiler answer?
3. What question does Nsight Systems answer?
4. What question does Nsight Compute answer?
5. Why might someone try Triton before handwritten CUDA?
6. What costs exist even for an almost-empty CUDA kernel?

Do not pause to understand every line of CUDA. The first pass is a map of the territory.

### Session 4 — Consolidate Lecture 1

#### Objective

Build a clear mental model of how a PyTorch GPU operation is launched and what each profiling or kernel tool is used for.

You are **not** required to write a Triton kernel, compile a custom CUDA extension, or operate Nsight Compute in this session.

#### Learn

Create this table in your own notes and explain every row in your own words:

| Tool or mechanism | What you must understand |
|---|---|
| PyTorch tensor operation | A high-level operation that may dispatch one or more CPU or GPU kernels |
| CUDA kernel | A function that executes parallel work on the GPU |
| `torch.utils.cpp_extension.load_inline` | A way to compile and load custom C++/CUDA code into PyTorch |
| Triton | A higher-level language and compiler for writing GPU kernels |
| PyTorch Profiler | Connects PyTorch operators with CPU activity and CUDA execution |
| Nsight Systems | Displays the wider CPU/GPU timeline, including launches, copies, kernels, and idle gaps |
| Nsight Compute | Examines an individual kernel using detailed hardware metrics |

#### Practical work

Run only these Lecture 1 examples if your environment permits:

1. Plain PyTorch square.
2. PyTorch Profiler square.

For the Triton and `load_inline` examples:

- Read the code.
- Identify where the input tensor enters.
- Identify where the GPU implementation is defined.
- Identify how the result returns to PyTorch.
- Do not spend time fixing compiler, CUDA-toolkit, or environment problems.

#### Assignment M2.1

Answer these questions without notes:

1. When `y = x.square()` runs on a CUDA tensor, what work happens on the CPU and what happens on the GPU?
2. What is a CUDA kernel launch?
3. Why can many small GPU operations be slower than one fused operation?
4. When would you use PyTorch Profiler?
5. When would you move from PyTorch Profiler to Nsight Systems?
6. When would you move from PyTorch Profiler to Nsight Compute?
7. What problem do Triton and `load_inline` each solve?

#### Deliverable

Create a short `lecture1-notes.md` containing:

- The tool table.
- A CPU-to-GPU execution sketch.
- Your answers to the seven questions.
- One profiler screenshot or the top-operations table from the square example.

#### Exit criteria

- You can explain the purpose of every tool in the table.
- You can describe the basic path from a PyTorch call to GPU execution.
- You understand that writing and tuning custom kernels is deferred until the later GPU-programming phase.

---

## Stage 2 — GPU architecture

### Session 5 — Lecture 4 and the GPU mental model

#### Objective

Understand enough GPU architecture to reason about why an operation may be limited by computation, memory movement, kernel-launch overhead, or input preparation.

#### Part A — Pre-read

Before watching Lecture 4, learn this hierarchy:

```text
CPU program
  -> launches a GPU kernel
GPU
  -> contains streaming multiprocessors (SMs)
SM
  -> schedules thread blocks and warps
Thread block
  -> group of cooperating threads assigned to one SM
Warp
  -> 32 NVIDIA GPU threads scheduled together
Thread
  -> one logical execution instance with per-thread state
```

Learn the memory hierarchy conceptually:

```text
Registers
  -> fastest and smallest; private to a thread
Shared memory / on-chip SRAM
  -> fast and shared by threads in a block
L1/L2 caches
  -> hardware-managed reuse
Global GPU memory
  -> large device memory containing tensors
CPU/host memory
  -> outside the GPU, reached over an interconnect
```

Do not memorize architecture-specific capacities or latency numbers.

#### Part B — Watch GPU MODE Lecture 4

Focus on this causal chain:

```text
Operation and tensor shape
  -> amount of arithmetic and data movement
  -> one or more kernel launches
  -> blocks and warps execute on SMs
  -> data moves through the memory hierarchy
  -> one resource becomes the main limitation
```

Pay attention to the lecture's examples of:

- Kernel-launch overhead.
- Unfused versus fused elementwise work.
- Global memory versus on-chip memory.
- Tiled matrix multiplication.
- Occupancy.

You only need conceptual understanding this month. Do not implement or tune tiled matrix multiplication.

#### Part C — Required concepts

| Concept | Month 2 understanding |
|---|---|
| SM | A GPU processing unit that schedules and executes blocks and warps |
| Warp | 32 threads scheduled together on NVIDIA GPUs |
| Registers | Very fast storage private to a thread |
| Shared memory | Fast on-chip memory shared by threads in one block |
| Global memory | Large device memory used for tensors, parameters, and activations |
| Memory bandwidth | Rate at which data can move between memory and compute units |
| Compute-bound | Arithmetic throughput is the main limitation |
| Memory-bound | Data movement is the main limitation |
| Launch-bound | Kernel-launch and dispatch overhead dominate because operations are small |
| Input-bound | The GPU waits for the CPU, DataLoader, storage, or host-to-device transfer |
| Fusion | Combining operations to reduce kernel launches and intermediate memory traffic |
| Occupancy | How much potential warp capacity is occupied by active warps; not the same as performance |
| GPU utilization | A busy-time signal; not a direct measurement of useful arithmetic efficiency |

#### Assignment M2.2

Answer these questions:

1. Why are registers and shared memory faster but smaller than global memory?
2. Why might a simple elementwise operation be memory-bound?
3. Why can matrix multiplication perform more useful arithmetic per byte loaded than vector addition?
4. Why can operation fusion improve performance even if the mathematics is unchanged?
5. Why can many tiny kernels become launch-bound?
6. Why can a GPU show high utilization while still executing inefficiently?
7. How would a slow DataLoader appear differently from a slow convolution?

For each workload, make only a qualitative hypothesis:

| Workload | Likely first hypothesis |
|---|---|
| Vector addition over a large tensor | Memory-bound |
| Large matrix multiplication | Potentially compute-bound |
| Very small matrix multiplication | Potentially launch-bound |
| Long chain of elementwise operations | Memory- and launch-bound |
| CNN with a slow DataLoader | Input-bound |

These are hypotheses, not universal truths. Profiling is used to test them.

#### Deliverable

Create `gpu-architecture-notes.md` containing:

- The execution hierarchy.
- The memory hierarchy.
- Definitions of the required concepts.
- Answers to the seven questions.
- One paragraph explaining fusion.

#### Exit criteria

- You can draw the GPU execution and memory hierarchies without notes.
- You can distinguish compute-, memory-, launch-, and input-bound behavior at a basic level.
- You can explain why high GPU utilization does not prove high efficiency.

---

## Stage 3 — Mixed precision

### Session 6 — AMP foundations and trainer integration

#### Objective

Understand mixed precision and add a correct FP32/AMP switch to your existing CIFAR-10 trainer.

#### Part A — Floating-point concepts

Learn these concepts before running the AMP recipe:

| Concept | Required understanding |
|---|---|
| FP32 | Standard 32-bit floating-point format commonly used for training |
| FP16 | 16-bit format with less range than BF16; small gradients are more vulnerable to underflow |
| BF16 | 16-bit format with an exponent range similar to FP32 but fewer precision bits |
| Dynamic range | Span between the smallest and largest representable magnitudes |
| Precision | How finely nearby values can be distinguished |
| Overflow | A value is too large for the format and may become infinity |
| Underflow | A small nonzero value becomes zero or loses useful information |
| Mixed precision | Different operations use different floating-point formats according to performance and numerical needs |
| Autocast | PyTorch chooses an appropriate dtype for eligible operations inside a context |
| Accumulation precision | Products may use a lower precision while sums accumulate in a wider format |
| Gradient scaling | Multiplies the loss before FP16 backward to protect small gradients, then unscales before updating parameters |

#### Part B — Hardware check

Run:

```python
import torch

print("GPU:", torch.cuda.get_device_name(0))
print("Capability:", torch.cuda.get_device_capability(0))
print(
    "Native BF16:",
    torch.cuda.is_bf16_supported(including_emulation=False),
)
```

Use this rule:

- If native BF16 is supported, use `torch.bfloat16` for the strict curriculum experiment.
- If the GPU is a T4 or otherwise lacks native BF16 support, use `torch.float16` with `GradScaler`.
- Label the fallback experiment **FP32 versus FP16**, not FP32 versus BF16.
- Use the same physical GPU for both compared runs.

#### Part C — Complete the AMP recipe

Work through PyTorch's official Automatic Mixed Precision recipe once.

Record:

- GPU model.
- FP32 runtime from the recipe.
- AMP runtime from the recipe.
- Peak allocated memory if shown.
- AMP dtype.
- Whether `GradScaler` is enabled.
- Dtype observed for at least one eligible operation inside autocast.
- Dtype of the model's stored parameters.

Focus on:

- Autocast wrapping the forward pass and loss.
- Backward running outside the autocast context.
- `GradScaler` for FP16.
- The difference between model parameter dtype and operation/output dtype.
- Why small or CPU-bound workloads may see little speedup.

#### Part D — Integrate AMP

Add a precision switch to the existing trainer:

```python
amp_dtype = (
    torch.bfloat16
    if torch.cuda.is_bf16_supported(including_emulation=False)
    else torch.float16
)

use_amp = config.precision == "amp"
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

Do not call `.half()` or `.bfloat16()` on the entire model when using autocast.

#### Assignment M2.3

Run several batches in FP32 and AMP modes. Verify:

- Both modes complete forward and backward passes.
- The loss remains finite.
- Parameters receive gradients.
- The optimizer updates parameters.
- The parameter dtype remains what you expect.
- The AMP dtype is recorded.
- `GradScaler` is enabled only when needed.

Answer without notes:

1. What does autocast decide?
2. What does autocast not decide?
3. Why should backward happen outside the autocast context?
4. Why does FP16 commonly need gradient scaling?
5. Why does BF16 normally need less protection against underflow?
6. Why does gradient scaling not intentionally change the final parameter update?

#### Deliverable

- Updated trainer supporting `precision="fp32"` and `precision="amp"`.
- `amp-notes.md` containing the recorded values and answers.
- A short smoke-test log showing finite losses for both modes.

#### Exit criteria

- The same trainer runs in FP32 and AMP modes.
- You can explain autocast and gradient scaling from first principles.
- You have correctly selected BF16 or the documented FP16 fallback based on hardware support.

---

## Stage 4 — PyTorch profiling

### Session 7 — Profiler recipe and trainer instrumentation

#### Objective

Learn the minimum PyTorch Profiler workflow needed to capture and narrate Project A2.

#### Part A — Complete the profiler recipe

Work through the official PyTorch Profiler recipe. Focus on:

- CPU time versus CUDA time.
- Self time versus total time.
- PyTorch operator versus CUDA kernel.
- Number of calls and average time.
- `ProfilerActivity.CPU` and `ProfilerActivity.CUDA`.
- `record_shapes=True`.
- `record_function`.
- Exporting a Chrome/Perfetto JSON trace.
- Profiling a short steady-state window instead of an entire training run.

You may skip deep stack-trace and memory-allocation analysis unless needed to diagnose a problem.

#### Part B — Instrument the trainer

Add named regions:

```python
from torch.profiler import record_function

with record_function("data_loading"):
    images, targets = next(data_iter)

with record_function("host_to_device"):
    images = images.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)

with record_function("zero_grad"):
    optimizer.zero_grad(set_to_none=True)

with record_function("forward_and_loss"):
    with torch.autocast(
        device_type="cuda",
        dtype=amp_dtype,
        enabled=use_amp,
    ):
        logits = model(images)
        loss = criterion(logits, targets)

with record_function("backward"):
    backward_step(loss)

with record_function("optimizer_step"):
    optimizer_step()
```

Use an explicit iterator if you want DataLoader waiting to appear inside `data_loading`. In a normal `for images, targets in loader` loop, fetching occurs before the loop body.

#### Part C — Capture a smoke-test trace

Profile only a short window:

```python
activities = [
    torch.profiler.ProfilerActivity.CPU,
    torch.profiler.ProfilerActivity.CUDA,
]

with torch.profiler.profile(
    activities=activities,
    schedule=torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=3,
        repeat=1,
    ),
    on_trace_ready=lambda p: p.export_chrome_trace(
        "outputs/traces/smoke_test.json"
    ),
    record_shapes=True,
    with_stack=False,
) as profiler:
    for _ in range(5):
        run_one_training_step(...)
        profiler.step()
```

Open the exported trace in Perfetto or `chrome://tracing`.

#### Assignment M2.4

Locate these regions in the smoke-test trace:

- Data loading.
- Host-to-device transfer.
- Forward and loss.
- Backward.
- Optimizer step.
- CUDA kernels beneath at least one PyTorch operator.

Answer:

1. What is the difference between CPU time and CUDA time?
2. What is the difference between self time and total time?
3. Why should the final benchmark run without the profiler enabled?
4. Why should only a few steady-state steps be traced?
5. What does `record_shapes=True` add?

#### Deliverable

- Instrumented trainer.
- `smoke_test.json` that opens successfully.
- `profiler-notes.md` containing answers and at least five trace observations.

#### Exit criteria

- The trace opens correctly.
- All five custom training regions are visible.
- You can connect a PyTorch operator to CUDA activity beneath it.
- You can explain why profiler timing is not the clean benchmark result.

---

## Stage 5 — Project A2

### Session 8 — Build and train the FP32 baseline

#### Objective

Reach at least 70% CIFAR-10 test accuracy within 15 epochs using a small four-convolution CNN without pretrained weights.

#### Part A — Model

Use a straightforward architecture. Do not spend time designing a novel CNN.

```text
Input: 3 x 32 x 32
Conv 3 -> 64, 3x3, padding=1
BatchNorm -> ReLU
Conv 64 -> 64, 3x3, padding=1
BatchNorm -> ReLU -> MaxPool
Conv 64 -> 128, 3x3, padding=1
BatchNorm -> ReLU
Conv 128 -> 128, 3x3, padding=1
BatchNorm -> ReLU -> MaxPool
Adaptive average pooling
Linear 128 -> 10
```

Before training:

- Print the tensor shape after every major stage.
- Count the trainable parameters.
- Confirm the output shape is `[batch_size, 10]`.
- Save the initial model state before the first optimizer update.

#### Part B — Data

Use:

- Random crop with padding for training.
- Random horizontal flip for training.
- Tensor conversion and CIFAR-10 normalization.
- Deterministic test preprocessing without random augmentation.

Do not use the test set to select updates during an epoch.

#### Part C — Training

- Precision: FP32.
- Maximum epochs: 15.
- Target: at least 70% test accuracy.
- No pretrained weights.
- Record all hyperparameters.
- Save the final checkpoint.
- Record test accuracy and epoch time after every epoch.

Use one conventional optimizer and scheduler configuration. Do not run a broad hyperparameter search.

If the model is below 70%, check in this order:

1. Input normalization.
2. Training augmentation.
3. `model.train()` and `model.eval()` placement.
4. Learning rate.
5. Optimizer and scheduler.
6. Accidental `torch.no_grad()` usage.
7. Accuracy calculation.
8. Dataset split and labels.

#### Part D — Benchmark FP32

After warm-up, measure a steady-state section of the training loop without the profiler.

Collect:

- Median or mean step time.
- Epoch time.
- Images per second if convenient.
- GPU utilization using one documented method.
- Final test accuracy.
- GPU model and batch size.

CUDA execution is asynchronous. Use CUDA events or explicit synchronization around the timed section rather than an unsynchronized Python timer.

Example with CUDA events:

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
run_training_steps(...)
end.record()
torch.cuda.synchronize()

elapsed_ms = start.elapsed_time(end)
```

#### Assignment M2.5

Create the FP32 result record:

| Field | Value |
|---|---|
| GPU | Measured |
| Batch size | Measured |
| Epochs | Measured |
| Final test accuracy | Measured |
| Mean/median step time | Measured |
| Mean epoch time | Measured |
| Images/sec | Measured or optional |
| GPU utilization | Measured |

#### Deliverable

- Initial model state.
- Final FP32 checkpoint.
- FP32 training log.
- FP32 benchmark result.
- Evidence of at least 70% test accuracy within 15 epochs.

#### Exit criteria

- Accuracy is at least 70%.
- The benchmark excludes profiler overhead.
- CUDA timing accounts for asynchronous execution.
- The initial model state is preserved for the AMP run.

---

### Session 9 — Mixed-precision run and paired traces

#### Objective

Run the same experiment using AMP, measure the difference, and capture comparable FP32 and AMP profiler traces.

#### Part A — Controlled AMP run

Restore the same initial model state used by the FP32 run.

Keep these constant:

- GPU and software environment.
- Model architecture.
- Initial model weights.
- Dataset and transforms.
- Batch size.
- Optimizer.
- Learning-rate schedule.
- Epoch count.
- Logging frequency.
- Seed and data ordering as far as practical.

Change only the precision path.

#### Part B — Train and benchmark AMP

Collect the same measurements as FP32:

- Final test accuracy.
- Mean or median step time.
- Epoch time.
- Images per second if convenient.
- GPU utilization using the same method.
- Actual AMP dtype.

Calculate:

\[
\text{speedup} =
\frac{\text{FP32 step time}}
     {\text{AMP step time}}
\]

If AMP step time is larger, report a speedup below 1.0x rather than hiding the result.

#### Part C — Capture paired profiler traces

Capture:

- `outputs/traces/fp32_trace.json`
- `outputs/traces/amp_trace.json`

Use the same:

- GPU.
- Model.
- Batch size.
- Number of warm-up and active steps.
- Profiler options.
- Custom region names.

Enable `record_shapes=True` for both traces.

#### Part D — Inspect both traces

Answer:

1. Where does one training step begin and end?
2. How much time appears in data loading?
3. How much time appears in host-to-device transfer?
4. How much time appears in forward and loss?
5. How much time appears in backward?
6. How much time appears in the optimizer?
7. Which stage changed most under AMP?
8. Are visible CPU or input gaps present?
9. Did CUDA kernel durations or names change?
10. Is the profiler observation consistent with the clean benchmark?

Approximate trace-based comparisons are sufficient. Do not attempt a detailed Nsight Compute analysis.

#### Assignment M2.6

Create this paired result table:

| Metric | FP32 | AMP |
|---|---:|---:|
| Final test accuracy | Measured | Measured |
| Step time | Measured | Measured |
| Epoch time | Measured | Measured |
| Images/sec | Measured/optional | Measured/optional |
| GPU utilization | Measured | Measured |
| Actual dtype | FP32 | BF16 or FP16 |
| Speedup | 1.00x | Calculated |

#### Deliverable

- AMP checkpoint and training log.
- AMP benchmark result.
- One FP32 trace.
- One AMP trace.
- Completed paired result table.
- Written answers to the ten trace questions.

#### Exit criteria

- Both runs are comparable and use the same initial weights.
- Both traces open successfully.
- Both traces contain data loading, transfer, forward, backward, and optimizer regions.
- The measured speedup is calculated and reported honestly.

---

### Session 10 — Final analysis and oral defense

#### Objective

Produce a concise report and demonstrate that you can connect the measurements to the underlying mechanisms.

#### Part A — Write the report

Keep the report to approximately one or two pages, excluding code and trace files.

Use this structure:

```text
1. Environment
2. Model and data pipeline
3. Shared experiment configuration
4. FP32 result
5. Mixed-precision result
6. Benchmark comparison
7. Profiler observations
8. Why the speedup is not 2x
9. Limitations
```

Include:

- GPU model.
- PyTorch and CUDA versions.
- Whether BF16 is natively supported.
- Actual AMP dtype.
- Whether `GradScaler` was used.
- Model parameter count.
- Batch size.
- Optimizer, scheduler, and learning rate.
- Epoch count.
- Final accuracy.
- Step time.
- Epoch time.
- GPU utilization.
- Speedup ratio.

#### Part B — Explain the non-2x result

Your explanation should connect your measurements to these ideas:

- Autocast does not convert every operation to BF16 or FP16.
- Some numerically sensitive operations remain in FP32.
- Model parameters and optimizer state may remain in FP32.
- Data loading and CPU transforms do not become twice as fast.
- Host-to-device transfer does not automatically become twice as fast.
- Kernel-launch overhead remains.
- The optimizer may receive little benefit from Tensor Cores.
- A small CIFAR-10 CNN may not generate enough matrix work to saturate Tensor Cores.
- Halving the representation size does not halve every byte moved during the complete training step.
- End-to-end speedup is limited by the fraction of the step that AMP accelerates.

Do not merely list these reasons. Point to evidence from your traces wherever possible.

#### Part C — Trace narration

Open either trace and narrate:

1. The CPU preparing or retrieving a batch.
2. The input transfer to the GPU.
3. The forward pass.
4. Loss computation.
5. Backward execution.
6. The optimizer step.
7. At least one PyTorch operator and the CUDA kernels beneath it.
8. The largest visible bottleneck or idle gap.

#### Part D — Oral examination

Answer without notes:

1. Why is a GPU not simply a faster CPU?
2. What does the CPU do when a CUDA kernel is launched?
3. What is an SM?
4. What is a warp?
5. Why is shared memory useful when global memory already exists?
6. What is the difference between compute-bound and memory-bound work?
7. What is launch-bound work?
8. Why can GPU utilization be high while execution remains inefficient?
9. What does autocast decide?
10. Why does FP16 commonly need gradient scaling?
11. Why does BF16 normally have less underflow risk than FP16?
12. Why must CUDA timing account for asynchronous execution?
13. Why is clean benchmark timing separated from profiler timing?
14. How would a DataLoader bottleneck appear in a trace?
15. Why was your measured AMP speedup not 2x?

#### Deliverable

- `report.md`.
- FP32 and AMP result table.
- Two profiler traces.
- Oral explanation completed without notes.

#### Exit criteria

- The report describes the actual experiment rather than expected results.
- Every performance claim is tied to a measurement.
- You can narrate the trace.
- You can explain the non-2x result from first principles.

---

## Acceptance checklist

Month 2 is complete when all of these are true:

- [ ] Session 0 parity harness passes all five cases across 20 seeds.
- [ ] The existing CIFAR-10 trainer has been refactored and checkpoint-tested.
- [ ] The hardware and software environment has been recorded.
- [ ] GPU MODE Lecture 1 has been watched.
- [ ] You can explain PyTorch Profiler, Nsight Systems, Nsight Compute, Triton, and `load_inline` at a high level.
- [ ] GPU MODE Lecture 4 has been watched.
- [ ] You can explain kernels, SMs, warps, registers, shared memory, cache, and global memory.
- [ ] You can distinguish basic compute-, memory-, launch-, and input-bound behavior.
- [ ] The official AMP recipe has been completed.
- [ ] The trainer supports FP32 and AMP modes.
- [ ] The actual AMP dtype and BF16 hardware support have been recorded.
- [ ] FP16 uses `GradScaler` when applicable.
- [ ] The official profiler recipe has been completed.
- [ ] The trainer contains named data, transfer, forward, backward, and optimizer regions.
- [ ] A four-convolution CNN reaches at least 70% CIFAR-10 test accuracy within 15 epochs.
- [ ] FP32 and AMP runs begin with the same initial model weights.
- [ ] Clean benchmark timing excludes profiler overhead.
- [ ] CUDA timing handles asynchronous execution correctly.
- [ ] Step time and GPU utilization are recorded for both runs.
- [ ] The FP32-to-AMP speedup ratio is calculated.
- [ ] One FP32 and one AMP trace open successfully.
- [ ] Both traces use `record_shapes=True`.
- [ ] You can identify data loading, transfer, forward, backward, and optimizer work in the traces.
- [ ] You can explain why the measured speedup is not 2x.
- [ ] The final report is complete.

---

## Glossary

### GPU execution

| Term | Meaning |
|---|---|
| Host | The CPU and its memory; prepares work and launches GPU operations |
| Device | The GPU and its memory |
| CUDA kernel | Function that executes parallel work on the GPU |
| Kernel launch | CPU/runtime request that queues a kernel for GPU execution |
| Grid | All thread blocks launched for one kernel |
| Thread block | Group of cooperating threads assigned to one SM |
| Thread | One logical execution instance of a kernel |
| Warp | Group of 32 NVIDIA GPU threads scheduled together |
| SM | Streaming multiprocessor that schedules blocks and warps and contains execution resources |
| SIMT | Execution model in which threads in a warp follow a shared instruction stream while retaining individual state |
| Occupancy | Proportion of potential warp capacity populated by active warps; not equivalent to speed |

### Memory and performance

| Term | Meaning |
|---|---|
| Register | Fast, per-thread on-chip storage |
| Shared memory/SRAM | Fast on-chip memory shared by threads in one block |
| Cache | Hardware-managed storage for recently or repeatedly accessed data |
| Global/device memory | Large off-chip GPU memory containing tensors, parameters, and activations |
| HBM | High-bandwidth device memory used on some GPUs; not all GPU memory is HBM |
| Memory bandwidth | Number of bytes that can move per unit time |
| Latency | Delay before an operation or data request completes |
| Throughput | Amount of work completed per unit time |
| Compute-bound | Arithmetic execution is the main limiting resource |
| Memory-bound | Data movement is the main limiting resource |
| Launch-bound | Kernel-launch and dispatch costs dominate useful execution |
| Input-bound | GPU waits for data preparation, storage, or host-to-device transfer |
| Fusion | Combining operations to reduce launches and intermediate memory traffic |
| Tiling | Dividing data into reusable blocks that fit in faster memory; detailed implementation is deferred |
| GPU utilization | Fraction of sampled time during which GPU kernels were active; not a complete efficiency metric |

### Precision

| Term | Meaning |
|---|---|
| FP32 | 32-bit floating-point format commonly used as default training precision |
| FP16 | 16-bit floating-point format with less exponent range than BF16 |
| BF16 | 16-bit format with FP32-like exponent range and fewer precision bits |
| Mixed precision | Use of multiple floating-point formats during one computation |
| Tensor Core | Specialized GPU hardware for matrix-like low-precision operations |
| Autocast | PyTorch mechanism that selects an operation-specific dtype inside a context |
| Accumulation precision | Format used when summing intermediate products |
| Dynamic range | Span between smallest and largest representable magnitudes |
| Precision | Ability to distinguish nearby values |
| Overflow | Value exceeds the format's representable range |
| Underflow | Tiny value becomes zero or loses useful representation |
| Gradient scaling | Temporarily scales the loss to protect FP16 gradients, then unscales before the optimizer update |
| NaN/Inf | Non-finite values indicating invalid arithmetic or overflow |

### Profiling

| Term | Meaning |
|---|---|
| Benchmark | Controlled measurement used to compare performance |
| Profile | Detailed observation used to find where time or memory is spent |
| Warm-up | Unmeasured iterations that absorb initialization and stabilization effects |
| Synchronization | Waiting for queued GPU work to finish |
| CUDA event | Device-timestamped marker used to measure GPU execution time |
| Trace | Time-ordered record of CPU operations, CUDA calls, transfers, and kernels |
| `record_function` | Adds a named region to a PyTorch Profiler trace |
| `record_shapes` | Records input shapes associated with profiled operators |
| CPU time | Host-side time associated with an event |
| CUDA time | Time associated with GPU kernels or device activity |
| Self time | Time directly spent in an event, excluding child events |
| Total time | Time including nested or child events |
| PyTorch Profiler | Framework-level profiler connecting operators to CPU and CUDA activity |
| Nsight Systems | System-level CPU/GPU timeline profiler |
| Nsight Compute | Kernel-level hardware-counter profiler |
| Perfetto | Browser-based viewer for Chrome JSON traces |

---

## Learning materials

Use these materials in order:

1. GPU MODE Lecture 1: **Profiling and Integrating CUDA Kernels in PyTorch**  
   <https://github.com/gpu-mode/lectures/tree/main/lecture_001>
2. GPU MODE Lecture 4: **Intro to Compute and Memory Architecture**  
   <https://github.com/gpu-mode/lectures/tree/main/lecture_004>
3. NVIDIA CUDA Programming Guide, programming model—reference only when Lecture 4 terminology is unclear  
   <https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html>
4. PyTorch Automatic Mixed Precision recipe  
   <https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html>
5. PyTorch AMP examples and reference  
   <https://docs.pytorch.org/docs/stable/notes/amp_examples.html>  
   <https://docs.pytorch.org/docs/stable/amp.html>
6. PyTorch Profiler recipe  
   <https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html>
7. Perfetto trace viewer  
   <https://ui.perfetto.dev/>

---

## Deferred work

Do not delay Month 3 for these topics. They return in later GPU-programming and inference-engineering phases:

- Writing Triton kernels from scratch.
- Writing C++/CUDA extensions using `load_inline`.
- Deliberately breaking custom CUDA bindings.
- Detailed Nsight Systems workflows.
- Detailed Nsight Compute hardware-counter analysis.
- Roofline calculations and exact arithmetic-intensity analysis.
- Tiled matrix-multiplication implementation and tuning.
- Occupancy calculation and tuning.
- Shared-memory bank conflicts.
- Memory coalescing optimization.
- Precision microbenchmark sweeps across many matrix shapes.
- Tensor-Core tile and dimension tuning.

---

## Final first-principles explanation

Before moving to Month 3, explain this chain without notes:

> The CPU launches GPU kernels. Those kernels execute blocks of threads on SMs, with threads scheduled in warps. The kernels read parameters and activations through the GPU memory hierarchy. Autocast allows suitable operations to use lower-precision values, reducing some memory traffic and enabling specialized hardware. PyTorch Profiler shows which parts of the training step benefit. Because only part of the total step is accelerated, end-to-end speedup is lower than the theoretical low-precision hardware ratio.

If you can explain this chain, reach the accuracy target, show both traces, state the measured speedup, and explain the result, Month 2 has achieved its purpose.