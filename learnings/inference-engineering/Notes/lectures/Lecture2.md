# Lecture 2 — How the GPU Runs Your Code, and What Mixed Precision Really Buys You

This lecture compiles all of Month 2 (Sessions 5–10) into one top-to-bottom read. Every script and notebook in `Month 2/` is used.

| Part | Topic | Source files |
|---|---|---|
| 1 | The GPU execution model | Session 5 notes |
| 2 | What limits a kernel: compute, memory, launch, input | Session 5 + oral report |
| 3 | Floating-point formats: FP32, FP16, BF16 | [7_floating_comparison.py](../../Month%202/7_floating_comparison.py) |
| 4 | Autocast and GradScaler | [2_amp.py](../../Month%202/2_amp.py), [2_amp.ipynb](../../Month%202/2_amp.ipynb) |
| 5 | Timing a GPU correctly | [2_amp.py](../../Month%202/2_amp.py), [4_cifar_fp32_baseline.py](../../Month%202/4_cifar_fp32_baseline.py) |
| 6 | The PyTorch profiler | [3_profiler.py](../../Month%202/3_profiler.py), [3_profile.ipynb](../../Month%202/3_profile.ipynb) |
| 7 | A real workload: the CIFAR-10 CNN | [4_cifar_fp32_baseline.py](../../Month%202/4_cifar_fp32_baseline.py), [5_cifar_amp.py](../../Month%202/5_cifar_amp.py) |
| 8 | Profiling FP32 against AMP, stage by stage | [6_fp32_vs_amp.py](../../Month%202/6_fp32_vs_amp.py) |
| 9 | Precision modes on a small model across devices | [1_cifar10.py](../../Month%202/1_cifar10.py) + `cifar/` package + `output/*.txt` |
| 10 | Why AMP did not give 2x | Session 10 report |
| 11 | Self-check | Oral report |

The thread through all of it: **to make something faster you first have to know what is limiting it.** Precision is one lever. It only helps if the part you speed up is a large share of the step.

---

## Part 1 — The GPU execution model

![](../images/gpu%20work%20flow.png)

### 1.1 From `a + b` to silicon

You write `result = a + b` with both tensors on the GPU. Four things happen:

1. **Python and PyTorch (CPU).** The **ATen** C++ dispatcher checks the tensors' device and picks the CUDA implementation of `add`.
2. **Launch configuration.** ATen decides how much work there is and shapes it into a **grid of thread blocks**.
3. **Asynchronous launch.** The CPU enqueues the kernel on a CUDA **stream** and moves on to the next Python line. It does not wait for the GPU.
4. **GPU execution.** The GPU hands blocks to its **SMs** (streaming multiprocessors).

```python
import torch
a = torch.randn(10_000_000, device="cuda")
b = torch.randn(10_000_000, device="cuda")
c = a + b            # returns almost instantly: the kernel is only *enqueued*
torch.cuda.synchronize()   # now the CPU actually waits for the GPU to finish
```

The asynchronous launch matters again in Part 5. It is why naive timing lies, and why a stray `.item()` is so expensive.

### 1.2 The hardware hierarchy

```
STAGE 1 — CPU / HOST
  result = a + b -> ATen dispatcher -> grid/block size -> ASYNC launch
        │  a "grid" of blocks
        ▼
STAGE 2 — GPU (a chip full of SMs)
  Grid: [Block0][Block1]...[Block999]  (queue)
  Does a block fit an SM? threads <= 1024? registers? shared mem?
     NO  -> launch FAILS (an error, not a queue)
     YES but SMs busy -> block WAITS until a slot frees up
        │  block assigned -> stays on ONE SM for life
        ▼
STAGE 3 — INSIDE ONE SM
  Resident blocks, each split into warps of 32 threads
  Warp scheduler, every cycle: pick a READY warp, issue its next instruction
  A warp stalled on memory is skipped -> zero-cost switch = "latency hiding"
        │
        ▼
STAGE 4 — ONE WARP (32 threads, lockstep / SIMT)
  T0 T1 ... T31 execute the SAME instruction on their OWN data
```

| Term | What it is |
|---|---|
| **SM** | A self-contained execution cluster with its own math units, registers, shared memory and warp schedulers. An A100 has over 100. |
| **Thread block** | A programmer-defined group of cooperating threads. It is assigned to **one** SM and never migrates. |
| **Warp** | The hardware's scheduling unit: exactly **32 threads** from one block, executing in lockstep (SIMT). |
| **Thread** | One lane in a warp, working on its own slice of data. |

Three consequences follow:

- **Block size should be a multiple of 32.** A 100-thread block becomes 4 warps, the last with only 4 real threads and 28 masked-off lanes, so 28 lanes of capacity are wasted.
- **Warps in a block = `ceil(threads / 32)`.** A 256-thread block has `ceil(256/32) = 8` warps.
- **Latency hiding is why GPUs tolerate slow memory.** A DRAM access takes hundreds of cycles. While one warp waits, the scheduler instantly issues from another ready warp. This works only if enough warps are resident.

```python
import math
def warps_in_block(threads): return math.ceil(threads / 32)
warps_in_block(256), warps_in_block(100)   # (8, 4)
```

### 1.3 Why a GPU is not "a faster CPU"

A CPU spends its silicon making **one instruction stream** fast: deep caches, branch prediction, out-of-order execution. A GPU spends it on **throughput**, running thousands of simple threads at once. More lanes help with independent work, but they cannot shorten a sequential dependency chain.

---

## Part 2 — What limits a kernel?

After Session 5 you should classify any slow GPU workload into one of four limits:

| Limit | Meaning | How it shows up |
|---|---|---|
| **Compute-bound** | Arithmetic units are the bottleneck. | Big matmuls and convolutions. Faster math, for example lower precision, helps. |
| **Memory-bound** | Moving data is the bottleneck. | Elementwise ops (`a + b`, ReLU). Each element is read and written for very little math. More ALUs do not help. |
| **Launch-bound** | CPU-side submission overhead dominates. | Many tiny kernels. GPU idle gaps line up with CPU launch activity. |
| **Input-bound** | The GPU waits for data prep, transfer or sync. | A long DataLoader region on the CPU with a GPU idle gap. |

**Arithmetic intensity** (operations per byte moved) tells you which of compute and memory is the likelier limit.

### The memory hierarchy

| Memory | Who can access it | Speed |
|---|---|---|
| Registers | The owning thread | Fastest |
| Shared memory | Threads in the same block (on-chip, programmer-managed) | Very fast |
| Global memory (DRAM) | Any thread on any SM | Slow, large |

**Shared memory** is why tiled matrix multiplication works. Tiles of the input matrices are loaded into shared memory once and reused for many multiply-accumulates before the next tiles are loaded. That cuts the number of global-memory reads.

### Quick consequences

- **Fusion** helps because it removes launches and avoids writing and rereading intermediate tensors.
- **Low GPU utilization with a fast kernel** usually means the GPU finishes quickly and then waits for CPU preprocessing, transfers, synchronization or the next request.
- **High utilization does not mean efficient.** It only means a kernel was active. A memory-bound kernel running near peak bandwidth is efficient even with low arithmetic throughput.

---

## Part 3 — Floating-point formats

Everything about mixed precision comes down to how many bits go to the **exponent** (range) and how many to the **fraction** (precision). [7_floating_comparison.py](../../Month%202/7_floating_comparison.py) builds this from scratch.

### 3.1 The representation

```
value = (-1)^s × (1.fraction) × 2^(exponent − bias)        (normal numbers)
```

| Format | Sign | Exponent bits | Fraction bits | Bias |
|---|---:|---:|---:|---:|
| FP32 | 1 | 8 | 23 | 127 |
| FP16 | 1 | 5 | 10 | 15 |
| BF16 | 1 | 8 | 7 | 127 |

- **BF16 is FP32's exponent with a truncated fraction.** It is literally the top 16 bits of an FP32 number. The script's `bin_to_bf16_correct` pads 16 zero bits to decode it.
- **FP16 trades range for precision.** It has more fraction bits (10 against 7) but only 5 exponent bits.

### 3.2 Deriving the limits

The all-ones exponent is reserved for inf/NaN. The script turns this into four formulas:

```python
def float_limits(fmt):
    e, f, bias = fmt["exponent"], fmt["fraction"], fmt["bias"]
    max_exp = (2**e - 2) - bias                 # largest usable exponent
    max_val       = (2 - 2**-f) * 2**max_exp    # 1.111…1 × 2^max_exp
    min_normal    = 2.0 ** (1 - bias)           # exponent field = 1
    min_subnormal = 2.0 ** (1 - bias - f)       # exponent field = 0, fraction = 0…01
    return max_val, -max_val, min_normal, min_subnormal
```

The script asserts these against `torch.finfo(dtype)`, and they match:

| | max | smallest normal | smallest subnormal |
|---|---:|---:|---:|
| FP32 | 3.40e+38 | 1.18e-38 | 1.4e-45 |
| FP16 | **6.55e+04** | **6.10e-05** | **5.96e-08** |
| BF16 | 3.39e+38 | 1.18e-38 | 9.2e-41 |

**This is the key to gradient scaling.** Anything smaller than about 6e-8 becomes exactly 0 in FP16 (**underflow**). Anything above 65,504 becomes inf (**overflow**). BF16 shares FP32's range, so it essentially never underflows where FP32 would not. In exchange it has only 7 fraction bits, so it rounds more coarsely.

Underflow and rounding are different problems. FP16 suffers underflow. BF16 suffers rounding error.

### 3.2b The exponent works on both sides of the decimal point

It is easy to think of the exponent as "how big the number is", meaning the digits before the point. It is not. The exponent is a **binary-point shifter**. It slides the point left or right, so it controls the size of numbers *below 1* just as much as above it. The exponent is stored with a bias, so it can be negative:

```
value = 1.fraction × 2^(exponent − bias)

  exponent − bias = +5   ->  1.25 × 2^+5 = 40          (point moved 5 places right: big number)
  exponent − bias =  0   ->  1.25 × 2^0  = 1.25
  exponent − bias = −3   ->  1.25 × 2^−3 = 0.15625     (point moved 3 places left: small fraction)
  exponent − bias = −24  ->  1.25 × 2^−24 ≈ 7.5e-8      (a typical tiny gradient)
```

So "range" means both ends: the largest numbers (positive exponents) and how close to zero you can get (negative exponents). The fraction bits only decide how finely you can resolve values *within* one `[2^k, 2^(k+1))` step.

| Format | Exponent range (after bias) | Smallest normal | Largest |
|---|---|---:|---:|
| FP16 (5 bits) | about −14 … +15 | 6.1e-05 | 6.5e+04 |
| BF16 (8 bits) | about −126 … +127 | 1.2e-38 | 3.4e+38 |

FP16's whole exponent window is only about 30 binary steps wide. It cannot go much below ~6e-8 or above 65,504. BF16 has the same window as FP32.

**Same numbers, two formats** (run with `torch.tensor(v).half()` / `.bfloat16()`):

| Value | FP16 | BF16 | What it teaches |
|---|---:|---:|---|
| `1e-8` (a tiny gradient) | **0.0** | 1.0012e-08 | FP16 **underflows**, because its exponent cannot go that negative. BF16 keeps it, with about 3 digits of accuracy. |
| `70000` (large activation) | **inf** | 70144 | FP16 **overflows**. BF16 survives, with coarse rounding. |
| `0.15625` | 0.15625 | 0.15625 | Exactly `1.25 × 2^-3`, so both store it exactly. The exponent handles the "after the point" part. |
| `1.003` | 1.0029 | **1.0** | BF16 has only 7 fraction bits, so its steps near 1 are `2^-7 ≈ 0.0078` wide. 1.003 rounds to 1.0. FP16's steps are `2^-10 ≈ 0.001`, so it keeps it. |
| `300.3` | 300.25 | 300.0 | Same effect at a larger scale: BF16 steps are 2.0 wide near 300, FP16 steps are 0.25. |

Read this table in two halves:

- **Rows 1–2 are an exponent (range) failure.** FP16 cannot reach that far in either direction. This is why FP16 needs GradScaler, which shifts tiny gradients up into FP16's window.
- **Rows 4–5 are a fraction (precision) failure.** BF16 reaches everywhere but blurs nearby values together. This is why BF16 needs no scaler, but is coarser per step.

A short way to remember it: **FP16 has the sharper ruler but a short tape. BF16 has the long tape but a blurrier ruler.**

### 3.3 Decoding raw bits

The script reinterprets integer bit patterns as floats with NumPy views, for example `np.array([val], dtype=np.uint16).view(np.float16)`. These are the extreme bit patterns it prints:

```
FP16 max           : 0 11110 1111111111  -> 6.5504e+04
FP16 min normal    : 0 00001 0000000000  -> 6.1035e-05
FP16 min subnormal : 0 00000 0000000001  -> 5.9605e-08
BF16 max           : 0 11111110 1111111  -> 3.3895e+38
```

**Subnormals** (exponent field 0) extend range downward by dropping the implicit leading 1. They give gradual underflow, at the cost of precision.

### 3.4 Where do the representable numbers live?

The script counts values in intervals. Positive floats sort the same as their bit patterns read as integers, so the count of values below `x` is simply the integer value of `x`'s bit pattern.

| Interval | FP32 | FP16 | BF16 |
|---|---:|---:|---:|
| `[0, 1)` | 1,065,353,216 (24.8% of all patterns) | 15,360 (23.4%) | 16,256 (24.8%) |
| `[0, 0.125)`, as a share of `[0, 1)` | 97.6% | 80.0% | 97.6% |
| `[0.5, 1)` | 2²³ = 8,388,608 | 2¹⁰ = 1,024 | 2⁷ = 128 |
| `[0.25, 1)` | 2·2²³ | 2·2¹⁰ | 2·2⁷ |

Two takeaways:

1. **Every binade `[2^k, 2^(k+1))` holds exactly `2^fraction` values.** That is the *resolution* of the format. BF16 has only 128 distinct values between 0.5 and 1, against 1,024 for FP16.
2. **A quarter of all bit patterns lie in [0, 1).** Floats are densest near zero, which is also where gradients live. This is why representing very small numbers (range) is the issue for gradients, and why FP16 struggles there.

---

## Part 4 — Mixed precision in PyTorch

### 4.1 Two separate mechanisms

| | Decides | Does not decide |
|---|---|---|
| **`torch.autocast`** | The **execution dtype** of each eligible op in the forward pass and loss. Matmul and conv go to FP16/BF16. Numerically sensitive ops (softmax, losses, reductions) stay FP32. | Gradient scaling, unscaling, or underflow/overflow protection. It also does not permanently convert the model. Parameters stay FP32. |
| **`GradScaler`** | Multiplies the loss by a scale `S` before backward, unscales gradients before the optimizer step, skips the step on inf/NaN gradients, and adapts `S`. | Which ops run in which dtype. |

Autocast uses a predefined per-device policy. It does not test whether a given input is safe in FP16. The classic example of why some ops stay FP32 is accumulation:

```text
total = a large value
add a very small value to total      # needs FP32 precision, or the small value vanishes
repeat many times
```

### 4.2 Why backward is outside the autocast block

Autocast picks dtypes during the **forward** pass. Autograd records those choices, and backward ops mirror them. Backward needs no autocast context of its own, and PyTorch recommends running it outside. Leaving the block does **not** make backward FP32. FP16 backward ops still run in FP16, which is exactly why they need scaling.

### 4.3 Why gradient scaling works

Gradients can be tiny, below FP16's 6e-8 floor. Scaling by a constant `S` shifts them into representable range, and unscaling recovers the exact gradient:

$$\frac{\partial (S\,L)}{\partial \theta} = S\,\frac{\partial L}{\partial \theta}, \qquad \frac{S\,\partial L/\partial\theta}{S} = \frac{\partial L}{\partial \theta}$$

It protects small gradients during backward. It does not change the effective learning rate. Floating-point rounding means results need not be bit-identical.

**BF16 normally needs no GradScaler** because its exponent range matches FP32. Underflow is rare but not impossible.

### 4.4 The experiment: [2_amp.py](../../Month%202/2_amp.py)

Setup: a 3-layer MLP (`4096 → 4096 → 4096 → 4096`, ReLU between), `batch_size=512`, 50 synthetic batches, `MSELoss`, `SGD(lr=0.001)`.

```python
def build_model(in_size, out_size, num_layers):
    layers = []
    for _ in range(num_layers - 1):
        layers += [torch.nn.Linear(in_size, in_size), torch.nn.ReLU()]
    layers.append(torch.nn.Linear(in_size, out_size))
    return torch.nn.Sequential(*layers)
```

**Run 1: FP32 baseline on CPU.** This is the reference, with no autocast and no scaling.

```python
logits = model(tx); loss = loss_fn(logits, ty)
loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
```
→ **3.514 s** for 5 epochs.

**Run 2: autocast FP16 on CPU, isolating `backward()`.** Forward and loss each sit inside their own autocast block, and `loss.backward()` sits outside, timed on its own.
→ **0.0017 s**. (A comment in the file says this was once "about 42 seconds" on the laptop. CPU FP16 matmul paths can be pathologically slow, so CPU timings for FP16 are not meaningful.)

**Run 3: autocast FP16 on CPU, full loop, no scaler.**
→ **3.306 s**, barely faster than 3.514 s. The speedup from low precision comes from **Tensor Cores on a GPU**, not from the dtype itself. (The script wraps this run in a docstring marked "Don't run the below". The notebook ran it.)

**Run 4: autocast FP16 + GradScaler on CUDA, 1 epoch.** This is the production pattern:

```python
grad_scaler = torch.amp.GradScaler(device=device)
for tx, ty in zip(x, y):
    with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
        logits = model(tx)
        loss = loss_fn(logits, ty)

    scaled_loss = grad_scaler.scale(loss)
    scaled_loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    grad_scaler.step(opt)       # unscales grads internally, then opt.step()
    grad_scaler.update()        # adapts the scale factor
    opt.zero_grad(set_to_none=True)
```
→ **0.602 s** for 1 epoch, **1561 MB** peak GPU memory.

`enabled=use_amp` makes autocast a no-op when `False`, so there is no if/else in the loop.

> **Gotcha in Run 4.** `clip_grad_norm_` runs on the **still-scaled** gradients, so `max_norm=1.0` is effectively `1.0 / S`. The correct order is `scaler.scale(loss).backward()` → `scaler.unscale_(opt)` → `clip_grad_norm_` → `scaler.step(opt)` → `scaler.update()`. The timing result is unaffected, but fix this before using clipping in real training.

**Takeaway.** FP16 autocast on CPU barely helps (3.514 s → 3.306 s). On the GPU the loop is fast (0.602 s per epoch). The two mechanisms are independent, and on CUDA with FP16 you normally use both.

---

## Part 5 — Timing a GPU correctly

Because launches are asynchronous, `time.time()` around a GPU call usually measures only the launch. There are two correct patterns.

**Pattern A — synchronize around a wall-clock timer** (used in the 2_amp helpers):

```python
def start_timer():
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()   # the .py used reset_max_memory_allocated
        torch.cuda.synchronize()               # drain earlier queued work FIRST
    global start_time; start_time = time.time()

def end_timer(msg):
    if device == "cuda":
        torch.cuda.synchronize()               # wait for the measured work to finish
    print(f"{msg}: {time.time()-start_time:.3f}s")
    if device == "cuda":
        print(f"Peak memory: {torch.cuda.max_memory_allocated()/2**20:.1f} MB")
```

Synchronize **before** starting, to exclude earlier queued work. Synchronize **before stopping**, to include the measured work.

**Pattern B — CUDA events** (used in the CIFAR trainers). Events are timestamped on the GPU's own timeline:

```python
if device == 'cuda':
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
else:
    start = time.perf_counter_ns()

# ... run all batches for one epoch ...

if device == 'cuda':
    end.record()
    torch.cuda.synchronize()              # wait for the end event before reading it
    elapsed_ms = start.elapsed_time(end)
else:
    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
```

**Rules of thumb for benchmarking**

- Warm up first. The first iterations pay for CUDA context creation, cuDNN autotuning, lazy module loading and allocator growth.
- Do not benchmark with the profiler on (see Part 6).
- Change one thing at a time, and re-run an unprofiled benchmark to confirm any improvement.

---

## Part 6 — The PyTorch profiler

A single wall-clock number cannot tell you *where* time goes. The profiler can. Everything here is from [3_profiler.py](../../Month%202/3_profiler.py) and [3_profile.ipynb](../../Month%202/3_profile.ipynb), using a `resnet18` forward pass on a `5×3×224×224` input.

### 6.1 Naive timing, then the profiler

```python
t0 = time.perf_counter_ns()
logits = model(random_input)
t1 = time.perf_counter_ns()
print((t1 - t0) / 1e6, "ms")      # one number: no operator breakdown, no CPU/CUDA split
```

On CUDA this number can also be wrong because of async launches. The profiler version:

```python
from torch.profiler import profile, ProfilerActivity, record_function

activities = [ProfilerActivity.CPU]
if torch.cuda.is_available():
    activities += [ProfilerActivity.CUDA]

with profile(activities=activities, profile_memory=True, record_shapes=True) as prof:
    with record_function("inference"):
        logits = model(random_input)

prof.export_chrome_trace("trace.json")      # open in ui.perfetto.dev or chrome://tracing
```

| Option | Effect |
|---|---|
| `activities` | Which streams to record: CPU ops, CUDA kernels. |
| `record_function("name")` | Adds a **named region** to the table and trace. |
| `profile_memory=True` | Adds CPU/CUDA memory columns. |
| `record_shapes=True` | Records the input shapes of every op. |
| `export_chrome_trace` | Writes a timeline: launches, gaps and overlap, not just aggregates. |

### 6.2 Reading the table

```python
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
print(prof.key_averages().table(sort_by="cpu_time_total",      row_limit=20))
print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=20))
print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total", row_limit=20))
```

- `key_averages()` collapses events into one row per operator name, averaged over repeated calls.
- **Self time vs total time.** *Total* includes nested child calls. *Self* excludes them. `aten::conv2d` may have large total time and tiny self time because the work is in its children.
- **CPU time vs CUDA time.** CPU time is host work, mostly dispatch and launch for a GPU op. CUDA time is how long the kernels ran on the device. A row like `aten::cudnn_convolution` can show small CPU self-time and larger CUDA time beneath it. In the notebook's GPU run, a large share of CPU time went to one-off costs like `Runtime Triggered Module Loading` (58 calls) and lazy function loading. This is warm-up, not steady state (see 6.4).
- **`record_shapes` + `group_by_input_shape=True`** splits one operator into a row per input shape. It exposes odd shapes, such as a short last batch, that dominate cost.
- The `get_sort_by_keys(event)` helper filters `dir(event)` for names containing `time`, `memory` or `count`. It tells you which attributes are valid sort keys.

### 6.3 Named regions for a training step

`record_function` regions nest. A trace then shows phases rather than a flat list of `aten::*` ops:

```python
with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
    with record_function("pre_training_work"):
        x, y = build_data(512, 4096, 4096, 1)
        model = build_model(4096, 4096, 3)
        loss_fn = torch.nn.MSELoss()
        opt = torch.optim.SGD(model.parameters(), lr=0.001)

    with record_function("training_loop"):
        for epoch in range(2):
            with record_function("one_training_loop"):
                for tx, ty in zip(x, y):
                    with record_function("forward_pass"):
                        logits = model(tx); loss = loss_fn(logits, ty)
                    with record_function("backward_pass"):
                        loss.backward()
                    with record_function("update_grad"):
                        opt.step()
                    with record_function("zero_grad"):
                        opt.zero_grad(set_to_none=True)

prof.export_chrome_trace("training_loop.json")
```

`training_loop` has large total time but near-zero self time. The leaf regions carry the cost.

This synthetic data sits in Python lists with no DataLoader and no `.to(device)`, so there is no separate data-loading or host-to-device region. A real trainer would add both (Part 8). To see kernel time on the GPU, add `ProfilerActivity.CUDA`.

### 6.4 Profiling only a steady-state window

Profiling every step of a long run is slow and produces unreadable traces. Use a schedule:

```python
scheduler = schedule(skip_first=1, wait=1, warmup=1, active=1, repeat=4)

def trace_handler(prof):
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
    prof.export_chrome_trace(f"./tmp/trace_{prof.step_num}.json")

with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True,
             schedule=scheduler, on_trace_ready=trace_handler) as prof:
    for epoch in range(10):
        logits = model(random_input)
        prof.step()                   # advances the schedule's counter
```

- `skip_first=1` ignores the first step once. Then each cycle is `wait` (skip) → `warmup` (run but discard) → `active` (record). The cycle repeats `repeat` times.
- With these numbers over 10 steps, the **active** steps are **4, 7 and 10**. That matches `tmp/trace_4.json`, `trace_7.json` and `trace_10.json` in the folder.
- `prof.step()` drives the schedule. It counts calls to `step()`, not internal operations.
- `on_trace_ready` fires once per completed active window, so each gets its own table and trace.

### 6.5 Profiler Q&A (Assignment M2.4)

1. **CPU time vs CUDA time?** CPU time is host time (dispatch and launch for GPU ops). CUDA time is kernel time on the device, attributed back to the launching op.
2. **Self vs total time?** Total includes children. Self is the operator or region alone.
3. **Why benchmark without the profiler?** The profiler adds overhead (events, shapes, memory tracking) and can perturb scheduling. Profiled time tells you *where* time goes, not *how fast* training is.
4. **Why trace only a few steady-state steps?** Early steps include one-time costs (context, autotune, caches, allocator growth). Short traces also stay small enough to open.
5. **What does `record_shapes=True` add?** An `Input Shapes` column, and it enables `group_by_input_shape`.

---

## Part 7 — The real workload: a CIFAR-10 CNN

From here the experiments use [4_cifar_fp32_baseline.py](../../Month%202/4_cifar_fp32_baseline.py) (FP32) and [5_cifar_amp.py](../../Month%202/5_cifar_amp.py) (AMP). The two files are identical except for the AMP additions, shown in 7.4.

### 7.1 The model

Four conv blocks, global average pooling, and a linear head. Shapes are for an input of `3×32×32`.

```python
class cifar_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1),   nn.BatchNorm2d(64),  nn.ReLU())                  # 64×32×32
        self.conv_block2 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1),  nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2)) # 64×16×16
        self.conv_block3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())                  # 128×16×16
        self.conv_block4 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1),nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2)) # 128×8×8
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1, 1))                                                                # 128×1×1
        self.linear_block = nn.Sequential(nn.Flatten(start_dim=1), nn.Linear(128, 10))                                       # logits
```

**Parameter count: 262,218.** You can verify it by hand. Each conv has `in×out×3×3 + out` parameters, and each BatchNorm has `2×channels`:

| Layer | Parameters |
|---|---:|
| conv1 (3→64) + BN | 1,792 + 128 |
| conv2 (64→64) + BN | 36,928 + 128 |
| conv3 (64→128) + BN | 73,856 + 256 |
| conv4 (128→128) + BN | 147,584 + 256 |
| linear (128→10) | 1,290 |
| **Total** | **262,218** |

### 7.2 The data pipeline

```python
tranforms = v2.Compose([
    v2.ToImage(),
    v2.RandomHorizontalFlip(),
    v2.RandomCrop(size=(32, 32), padding=4),
    v2.ToDtype(torch.float32, scale=True),     # uint8 [0,255] -> float [0,1]
    v2.Normalize(mean=[0.5]*3, std=[0.5]*3),   # -> [-1, 1]
])
train = CIFAR10(root=CIFAR10_PATH, train=True,  download=CAN_DOWNLOAD, transform=tranforms)
test  = CIFAR10(root=CIFAR10_PATH, train=False, download=CAN_DOWNLOAD, transform=tranforms)
```

To plot an image, undo the normalization (`(img + 1) / 2`) and `permute(1, 2, 0)` from channels-first to channels-last.

> **Gotcha.** The **same** `tranforms` is passed to the *test* set, so `RandomHorizontalFlip` and `RandomCrop` also randomize test images. Evaluation should use only `ToImage → ToDtype → Normalize`. This adds noise to every accuracy number below. (The Session 8 note says the test set has no random transforms, but the code applies them.)

All of this runs on the **CPU** inside the DataLoader. Autocast cannot speed it up.

### 7.3 Training loop, checkpoints and evaluation

Training: `Adam(lr=0.001)`, `CrossEntropyLoss`, `batch_size=1024`, checkpoint after every epoch.

```python
def store_checkpoint(model, optim, epochs, file):
    torch.save({'model': model.state_dict(),
                'optim': optim.state_dict(),
                'epochs_done': epochs}, file)

def load_checkpoint(file, device, model, optim):
    ck = torch.load(file, map_location=device, weights_only=True)
    model.load_state_dict(ck['model'])
    if optim is not None:
        optim.load_state_dict(ck['optim'])
    return ck['model'], ck['optim'], ck['epochs_done']
```

A checkpoint holds the optimizer state too. Adam's moment buffers are needed to resume training faithfully. `FRESH_TRAINING = False` resumes from a saved file. A checkpoint is always saved in FP32, even after an AMP run.

Evaluation reloads the checkpoint into a fresh model, then:

```python
model.eval()                          # BatchNorm uses running stats
correct = 0
with torch.no_grad():                 # no autograd graph
    for tx, ty in test_loader:
        tx, ty = tx.to(device), ty.to(device)
        ypred = torch.argmax(model(tx), dim=1)
        correct += torch.eq(ypred, ty).int().sum().item()
accuracy = correct / test_n
```

### 7.4 What AMP changes

The only difference is the precision path, plus a `scaler` argument threaded through the loop and checkpoint:

```python
with autocast(device_type=device, dtype=torch.float16, enabled=True):
    logits = model(x)
    batch_loss = loss_fn(logits, y)

scaler.scale(batch_loss).backward()
scaler.step(optim)
scaler.update()
optim.zero_grad(set_to_none=True)
```

`logits.dtype` prints `torch.float16` under autocast. The forward and loss activations are FP16, while the parameters and the optimizer state are FP32. The AMP checkpoint also stores `scaler.state_dict()` (the current scale factor), so a resumed run does not restart scale discovery.

The AMP script also calls `set_seed(42)`, which seeds torch, Python and CUDA RNGs and sets `cudnn.deterministic=True` and `benchmark=False`. The FP32 baseline script does **not** call `set_seed`.

### 7.5 Results so far

| Run | Scope | Time | Accuracy |
|---|---|---:|---:|
| FP32 on CPU | 15 epochs | 87 s | 70% |
| FP32 on GPU + profiler | all batches | 47 s | n/a |
| FP32 on GPU | single batch | ~700 ms (includes warm-up) | n/a |
| FP32 on GPU | 15 epochs | 31 s (reproduced twice) | 58% |
| AMP (FP16) on GPU | single batch | 620–920 ms | n/a |
| AMP (FP16) on GPU | 1 epoch (see below) | 31 s logged | 68% |

What these show, and what they do not:

- **GPU vs CPU.** 31 s against 87 s is about 2.8x faster for the full run.
- **Profiler tax.** 47 s against 31 s is about 52% overhead. This is why you do not benchmark with the profiler on.
- **The single-batch time (700 ms) is dominated by warm-up.** It is not a per-batch steady-state number.
- **Accuracy is not comparable across these runs.** The initial weights and seeds differ (FP32 has no seed, AMP has `seed=42`), and the AMP run trained 1 epoch against FP32's 15. The 70% / 58% / 68% spread is run-to-run noise plus these differences, **not** a device or precision effect. The 68% after one epoch is surprising and was not investigated. Re-run both under identical seeds and epoch counts before drawing any conclusion.
- **The two "31 s" figures are not like-for-like.** The FP32 one covers 15 epochs. The AMP one is a per-epoch log line. Do not read it as "AMP is no faster".

---

## Part 8 — Profiling FP32 against AMP, stage by stage

[6_fp32_vs_amp.py](../../Month%202/6_fp32_vs_amp.py) isolates one training step on synthetic data (`10000×3×32×32` random input, random targets) so only precision changes. A flag `amp_enabled` switches between the two paths, and each phase gets a `record_function` region:

```python
with profile(activities=activities, profile_memory=True, record_shapes=True) as prof:
    with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=amp_enabled):
        with record_function('forward_pass'):
            logits = model(rand_input)
        with record_function('calculate_loss'):
            loss = loss_fn(logits, rand_target)

    if amp_enabled:
        with record_function('grad_scaler_backward'): scaler.scale(loss).backward()
        with record_function('update_gradients'):     scaler.step(optimizer)
    else:
        with record_function('normal_backward'):      loss.backward()
        with record_function('update_gradients'):     optimizer.step()

    with record_function('zero_grad'):
        optimizer.zero_grad(set_to_none=True)

prof.export_chrome_trace("cuda_simple_fp32_trace.json" if not amp_enabled else "cuda_simple_amp_trace.json")
```

The paired traces ([cuda_simple_fp32_trace.json](../../Month%202/cuda_simple_fp32_trace.json), [cuda_simple_amp_trace.json](../../Month%202/cuda_simple_amp_trace.json)) are loaded into Perfetto.

### 8.1 What a single step looks like (1024-batch trace of the AMP trainer)

| Stage | Time |
|---|---:|
| Host-to-device copy of `x` and `y` | ~3 ms |
| Forward | ~3.14 ms |
| Loss (`CrossEntropyLoss` on `[1024, 10]`) | ~147 µs |
| `batch_loss.item()` | **~18 ms** |
| `scaler.scale(loss).backward()` | ~3.49 ms |
| `scaler.step` + `update` | ~3.03 ms |

**The `.item()` trap.** `batch_loss.item()` costs 18 ms, about 120x the loss computation itself. `.item()` copies a value to the CPU, which forces the CPU to wait for every queued kernel to finish. That is a **synchronization point**. It drains the async pipeline, and the GPU then sits idle while Python issues the next batch. The fix is to accumulate the loss on the GPU and call `.item()` once per epoch:

```python
running = torch.zeros((), device=device)
for x, y in loader:
    ...
    running += batch_loss.detach()       # stays on the GPU
epoch_loss = (running / len(loader)).item()   # one sync per epoch
```

This is the most avoidable cost in the whole experiment, and it is unrelated to precision.

### 8.2 FP32 vs AMP (10,000-image batch)

The top table is **CPU time** per region, so mostly launch and submission cost. On the GPU the work is asynchronous. The bottom tables give actual GPU time.

| Stage (CPU submit time) | FP32 | AMP | Note |
|---|---:|---:|---|
| `forward_pass` | 5.682 ms | 5.635 ms | same |
| `calculate_loss` | 0.388 ms | 0.379 ms | same |
| `backward` | 3.583 ms | 4.039 ms | slightly slower under AMP (GradScaler bookkeeping) |
| `update_gradients` | 2.709 ms | 585 ms | 581.5 ms of this is one `.item()` sync (the scaler checks for inf/NaN), not a real AMP cost |
| `zero_grad` | 0.085 ms | 0.101 ms | same |

| GPU time | FP32 | FP16 |
|---|---:|---:|
| `forward_pass` | 372 ms | 215 ms |

| Dominant conv kernel | Time |
|---|---:|
| FP32: `_5x_cudnn_ampere_scudnn_128x64_relu_xregs_large_nn_v1` | 11.330 ms |
| FP16: `_5x_cudnn_ampere_fp16_scudnn_fp16_128x64_relu_medium_nn_v1` | 5.702 ms |

### 8.3 What changed

- **cuDNN picked a different kernel.** The FP16 run uses kernels with `fp16` in the name (`..._fp16_scudnn_fp16_...`, or `implicit_convolve_sgemm<__half,...>`) and a different tile and register configuration (`medium` against `large`). Autocast changes **which kernel runs**, not just how the same kernel reads its bits.
- **The convolution got about 2x faster** (11.33 → 5.70 ms, 1.99x). Forward GPU time dropped from 372 ms to 215 ms (1.73x).
- **Everything else stayed roughly the same.** Host-to-device copy (~3 ms), loss, and `zero_grad` did not change. Backward got slightly worse, and the optimizer step also got slightly worse.
- **The conversion kernels remain.** `nchwToNhwcKernel` / `nhwcToNchwKernel` layout kernels show up in both traces, so those launches are not removed by autocast.
- **The trace shows CPU gaps.** After the `.item()` sync the GPU finishes its queue and waits on Python. That is a clear input/launch-bound signature.

---

## Part 9 — Precision modes on a small model, across devices

The scripts so far used the 4-conv CNN. [1_cifar10.py](../../Month%202/1_cifar10.py) is an earlier, more structured experiment on a **LeNet-style** model, with helpers in the `cifar/` package. It compares FP32, FP16 and BF16 end to end.

### 9.1 The package

| File | Role |
|---|---|
| `cifar/env.py` | `env_check()` prints Python, PyTorch, CUDA build, GPU name, compute capability, BF16 support, cuDNN version. `set_seed()` seeds torch, Python and NumPy, plus CUDA when available, and makes cuDNN deterministic. |
| `cifar/configs.py` | `gpu_configs(precision_mode)` returns `(device, amp_type, use_amp, scaler)`. FP16 builds a `GradScaler`. BF16 requires CUDA with native support and builds **no** scaler. |
| `cifar/model.py` | LeNet-style CNN: `Conv(3→6,5)` → pool → `Conv(6→16,5)` → pool → `400→120→84→10`. |
| `cifar/dataloaders.py` | Builds the loaders. `pin_memory` is on when CUDA is available. |
| `cifar/training.py` | `train_one_epoch` and `train_loop`. Uses autocast and, when a scaler is enabled, the scale/step/update sequence. |
| `cifar/checkpoints.py` | Saves model, optimizer, scaler (or `None`), config and `epochs_run`. `verify_checkpoint` reloads it and asserts the logits match exactly. |
| `cifar/test.py` | `get_accuracy` under `no_grad`, using autocast with the same dtype as training. |

The device-specific policy in `gpu_configs` is worth reading:

```python
if precision_mode == "fp16":     # needs GradScaler
    amp_type = torch.float16; use_amp = True
    scaler = torch.amp.GradScaler(device=device, enabled=True)
elif precision_mode == "bf16":   # same range as FP32, so no scaler
    assert torch.cuda.is_bf16_supported(including_emulation=False)
    amp_type = torch.bfloat16; use_amp = True
```

Two details in `1_cifar10.py` to note: `MODE = 'NEW'` or `'RESUME'` chooses between fresh training and loading a checkpoint, and `config` is saved inside the checkpoint, so resume rebuilds the exact same setup.

### 9.2 Results

Config: seed 42, `batch_size=128`, SGD `lr=0.01`, 5 epochs then a resume for 5 more (10 total).

| Run | Device | Epoch time | Acc. @ 5 epochs | Acc. @ 10 epochs |
|---|---|---:|---:|---:|
| FP32 | CUDA (NVIDIA L4) | ~8.9 s | 34.58% | 45.01% |
| FP16 + GradScaler | CUDA (L4) | ~8.7 s | 34.63% | 44.98% |
| BF16 (no scaler) | CUDA (L4) | ~8.8 s | 34.77% | 45.02% |
| FP32 | MPS (Apple GPU) | ~9.6 s | 34.64% | 45.00% |
| FP16 | MPS (Apple GPU) | **~4.0 s** | 34.66% | 45.00% |

The `output/mps_bf16.txt` file is misnamed. Its header shows it was run on the **CUDA L4**, not MPS.

What to take from this:

- **Accuracy is essentially identical across all precisions**, within about 0.2 percentage points. With the same seed, the loss curves track each other to roughly the third decimal. Mixed precision here did not hurt training.
- **BF16 trains fine with no GradScaler**, matching the range argument from Part 3.
- **On the L4 there is no speedup at all** (8.7 to 8.9 s per epoch in every precision). This model is tiny: a few small convs and a few small linears at batch 128. At that size the time goes to kernel launches, the DataLoader (`workers=0`, with CPU-side transforms) and `.item()` syncs, not to Tensor Core math. Lower precision has nothing to accelerate.
- **On MPS, FP16 is about 2.4x faster** (9.6 s → 4.0 s). The device and its kernels decide whether the dtype matters. The same code gives a 2.4x gain on one backend and none on another.
- **The checkpoint round trip is exact.** The `True` / `tensor(0., ...)` lines in each output come from `verify_checkpoint`: the reloaded model's logits are identical.

This is Part 2 in action. When a workload is launch-bound or input-bound, speeding up the arithmetic cannot help.

---

## Part 10 — Why AMP did not give 2x

The headline finding of the Month 2 report is that two speedup numbers disagree:

| Measurement | Ratio |
|---|---:|
| Isolated convolution kernel (11.330 → 5.702 ms) | **1.99x** |
| Forward GPU time on the 10k batch (372 → 215 ms) | 1.73x |
| End-to-end step time (~700 ms vs 620–920 ms) | **0.76x–1.13x** (within noise) |

Why the kernel gain does not show up end to end:

1. **Autocast does not convert everything.** Only eligible ops (conv, matmul) run in FP16. Numerically sensitive ops like the softmax and log-sum-exp in `CrossEntropyLoss` stay in FP32, and the loss time did not change (0.388 → 0.379 ms).
2. **Parameters and optimizer state stay FP32.** Adam's moment updates are elementwise, not matmul or conv, so they get no Tensor Core benefit. The optimizer stage got slightly *slower* under AMP (GradScaler's unscale and inf-check add work).
3. **Data loading does not speed up.** `RandomCrop`, `RandomHorizontalFlip` and `Normalize` run on the CPU and are unaffected by the GPU's precision. This is a fixed floor under every batch.
4. **Host-to-device transfer does not shrink.** It stayed at ~3 ms. Images are read, augmented and copied as FP32 before the model is reached, and FP16 only exists inside the forward and loss region.
5. **Launch overhead remains.** The layout-conversion kernels and many small ops still launch, just on smaller tensors.
6. **The workload is small.** With `batch_size=1024` and channel widths of 64 and 128, there may not be enough matrix work to saturate Tensor Cores. That is also what Part 9 showed on the even smaller LeNet.
7. **Bytes moved do not all halve.** Inputs, parameters, gradients and optimizer state remain FP32.
8. **Sync points hide the gain.** `batch_loss.item()` (~18 ms per batch) is bigger than any saving from precision, and it was not removed before the benchmarks were taken.

### Amdahl's law

If a fraction `p` of the step is sped up by a factor `s`, the overall speedup is:

$$\text{speedup} = \frac{1}{(1-p) + p/s}$$

```python
def amdahl(p, s): return 1 / ((1 - p) + p / s)
amdahl(p=0.5, s=2)    # 1.33x: even if half the step becomes 2x faster
amdahl(p=0.2, s=2)    # 1.11x: the realistic regime for a small CNN
```

Even a perfect 2x on a convolution that is only 20% of a step gives about 1.11x. Once `.item()`, transfers, optimizer and data loading are counted, a result inside run-to-run noise is expected.

**A careful caveat on the numbers.** The per-stage table in Part 8 sums to about 12.45 ms for FP32 and 13.70 ms for AMP, excluding the `.item()` sync. Those are **CPU submission times**, not GPU execution times, so they do not add up to a step time. They show that no stage other than the convolution changed. A proper speedup calculation needs GPU time per stage from the CUDA rows of the trace, measured with the profiler off.

---

## Part 11 — Limitations and what to do next

From the Session 10 report:

- The AMP run covered 1 epoch against FP32's 15. **Do not compare their epoch times.** Only the single-batch range and the isolated-kernel numbers are reliable.
- Accuracies (58% / 68% / 70%) came from different seeds and epoch counts. Treat the gaps as variance, and re-run under identical conditions.
- **GPU utilization was never measured.** Run `nvidia-smi dmon` or `nvidia-smi --query-gpu=utilization.gpu` alongside a training run.
- The 2x comparison used a single synthetic batch with no DataLoader. It isolates stage costs well but is not an end-to-end benchmark.
- BF16 was never run on the 4-conv CNN, only on the small LeNet in Part 9.
- The `.item()` sync was identified but never removed before benchmarking.
- The test set receives random augmentation (Part 7.2).
- Run 4 in Part 4 clips gradients before unscaling.

**Next experiments, in priority order:**

1. Remove the per-batch `.item()`, and add `set_seed` to both scripts.
2. Fix the test transform, then run FP32 and AMP for the same number of epochs.
3. Benchmark with CUDA events and the profiler off, with a warm-up epoch discarded. Report per-epoch time.
4. Add BF16 to the 4-conv CNN, and measure GPU utilization.
5. Try a larger batch or wider channels to see whether the AMP speedup grows.

---

## Part 12 — Self-check (answer aloud, then compare)

**1. Why is a GPU not simply a faster CPU?**
A CPU optimizes the latency of one instruction stream, including dependencies and branches. A GPU optimizes throughput by running many threads at once. More lanes cannot shorten a sequential dependency chain.

**2. What does the CPU do when a CUDA kernel is launched?**
It runs host code that submits the kernel and arguments to a stream, then continues without waiting. Operations in one stream run in order. A synchronization, or something that needs a value on the CPU such as `.item()`, forces the CPU to wait.

**3. What is an SM, and what is a warp?**
An SM is a hardware unit with execution units, warp schedulers, registers and shared memory. A warp is 32 threads from the same block that the SM schedules together. They run the same instruction on their own data.

**4. How many warps are in a 256-thread block?**
`ceil(256 / 32) = 8`.

**5. Why use shared memory when global memory exists?**
It is fast, on-chip and shared by a block, so threads can reuse data and exchange intermediates instead of repeating global reads.

**6. Compute-bound vs memory-bound vs launch-bound?**
Compute-bound is limited by arithmetic throughput, memory-bound by data movement, launch-bound by kernel submission overhead (many tiny kernels). Arithmetic intensity, in operations per byte, guides the first two. For launch-bound work, look for short kernels and GPU gaps that line up with CPU launch activity.

**7. Why can utilization be high while execution is inefficient?**
Utilization only says kernels were active. They may do redundant work or use the hardware poorly. A memory-bound kernel near peak bandwidth can be efficient with low arithmetic throughput.

**8. What does autocast decide?**
The dtype of eligible ops, from a predefined per-device policy. It does not convert the model's parameters, scale gradients, or test whether each input is safe in FP16.

**9. Why does FP16 need gradient scaling?**
FP16 values below about 6e-8 underflow to zero, so small gradients vanish. Scaling the loss by `S` lifts the gradients into range. They are unscaled before the optimizer step.

**10. Why does BF16 underflow less?**
It has 8 exponent bits (like FP32) against FP16's 5, so it represents much smaller magnitudes. It pays with fewer fraction bits (7 against 10), so it rounds more coarsely.

**11. Why must CUDA timing account for async execution?**
A CPU timer can stop before the GPU finishes. Synchronize before starting (to exclude earlier work) and before stopping (to include the measured work), or use CUDA events and wait for the end event.

**12. Why separate clean benchmarks from profiler runs?**
Profiling adds overhead and can change behavior. Establish performance with warmed-up, synchronized, unprofiled runs. Use traces to explain bottlenecks. Verify any fix with another clean benchmark.

**13. How does a DataLoader bottleneck appear in a trace?**
As a long batch-retrieval region on the CPU overlapping a GPU idle gap. A long DataLoader region alone is not enough evidence, since it may overlap useful GPU work.

**14. Why was the measured AMP speedup not 2x?**
FP16 halved the convolution kernel (1.99x), but that is only part of a step. The FP32 loss, optimizer, data loading, transfers, launch overhead and sync points did not speed up, and GradScaler added a little work. By Amdahl's law the end-to-end gain is limited by the accelerated fraction, not by FP16 having half as many bits.
