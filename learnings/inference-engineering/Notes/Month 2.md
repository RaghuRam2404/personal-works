
## Session 5

[https://www.perplexity.ai/search/76fe7721-a448-4050-8309-7b3c3f026b7d](https://www.perplexity.ai/search/76fe7721-a448-4050-8309-7b3c3f026b7d)

### How the GPU works and in what order?

![](../images/gpu%20work%20flow.png)

```text
Here's the whole story stitched into one narrative, in plain language, pulling together everything we've built up.

## Where It Starts: You Write Normal Python Code

You write something like `result = a + b` where `a` and `b` are PyTorch tensors sitting on a GPU. Nothing about that line looks like "GPU programming" — it looks like ordinary math. But under the hood, PyTorch's C++ engine called **ATen** intercepts that call, checks which device the tensors live on, and picks the right low-level implementation to run it. If the tensors are on a GPU, ATen decides how much work needs doing and shapes that work into a **grid of thread blocks** — think of it as ATen writing a work order: "do this addition, split into N chunks, hand it to the GPU." Then it fires off the **kernel launch** and the CPU immediately moves on to its next line of code, because this launch is asynchronous — the CPU doesn't sit around waiting. [arxiv](https://arxiv.org/html/2603.12465v1)

## The GPU Is Not One Big Brain — It's Many Small Ones

A GPU chip is built from a bunch of identical mini-processors called **streaming multiprocessors (SMs)** — a big data-center GPU like the A100 has over a hundred of them. Each SM is fully self-sufficient: it has its own math units, its own scratchpad memory (called shared memory), and its own scheduler. This matters because it explains why a GPU is fundamentally different from a CPU. A CPU core is built to run one or two threads really fast, with a lot of clever logic dedicated to that single thread. An SM instead is built to run **thousands of simple threads at once**, sacrificing per-thread cleverness for sheer volume.

## Handing Out the Work: Thread Blocks

The work order from ATen — the grid — is chopped into pieces called **thread blocks**. Each block is a self-contained bundle of threads that cooperate with each other. The rule that governs everything downstream: once a thread block is handed to an SM, it stays glued to that one SM for its entire life — it never moves. But a block only gets assigned to an SM if that SM actually has enough capacity — enough registers, enough shared memory, enough thread slots — to hold the *whole* block at once. If a block asks for more than any single SM can ever provide (say, more than 1,024 threads, or too much shared memory), the launch fails outright before anything even runs. If there simply isn't room right now (the SM's busy with other blocks), the extra blocks just wait their turn in a queue until room frees up. [olcf.ornl](https://www.olcf.ornl.gov/wp-content/uploads/2013/02/GPU_Opt_Fund-CW1.pdf)

## Inside One SM: Warps Are the Real Unit of Work

Here's the part that feels counter-intuitive at first: an SM doesn't run threads one at a time, and it doesn't run an entire block as one giant lump either. It splits every resident block into fixed groups of exactly **32 threads**, called a **warp**. All 32 threads in a warp move in lockstep — every cycle, they all execute the exact same instruction together, just on their own private piece of data. This is called SIMT (single instruction, multiple threads).

If a block's thread count isn't a clean multiple of 32 — say 100 threads — the hardware still carves it into warps of 32 (4 warps: three full, one with only 4 real threads and 28 "masked off" idle lanes). Nothing crashes, but those 28 lanes are wasted work capacity, which is exactly why experienced CUDA programmers deliberately size blocks in multiples of 32. [stackoverflow](https://stackoverflow.com/questions/26611241/why-launch-a-multiple-of-32-number-of-threads-in-cuda)

## The Clever Trick: Hiding Slowness by Switching Warps Instantly

Once a block's warps are resident on an SM, they don't all execute at the same instant — the SM typically only has enough execution hardware to actively process a handful of warps per cycle, even though it can *hold* dozens resident at once. So the SM's **warp scheduler** looks, every single clock cycle, at all the warps sitting ready on that SM (possibly from several different blocks at once) and picks one that's ready to go, issuing it the next instruction.

Here's the payoff: if a warp is stuck waiting on something slow — like fetching data from memory, which can take hundreds of cycles — the scheduler doesn't let the SM sit idle. It instantly swaps to a *different* warp that's ready to run, with zero switching cost, because each resident warp already has its own registers sitting on-chip. This is called **latency hiding**, and it's the single biggest reason GPUs can be blazing fast despite memory being relatively slow: as long as there's enough other work resident to fill the gaps, the SM is almost never truly stalled.

## Zooming All the Way In: One Thread

At the very bottom, a single thread is just one lane inside a warp, running the kernel's instructions on its own slice of the data — its own index into the array, its own small piece of the math — while sharing access to that SM's local shared memory if the kernel is written to use it for cooperation between threads in the same block.

## Why This Whole Picture Matters

Every performance problem you'll debug later comes back to this chain: is the GPU running out of blocks that fit (occupancy problem), are warps stalling too often without enough other warps to hide the latency (not enough parallel work), or are warps wasting lanes because block sizes weren't multiples of 32 (inefficient sizing)? That's precisely the "four possible limits" — computation, memory movement, kernel-launch overhead, and input preparation — that Session 5's Lecture 4 is going to teach you to diagnose.
```

### Flow

```
STAGE 1 — CPU / HOST
┌────────────────────────────────────────────────────────────┐
│ result = a + b   (your Python/PyTorch line)                │
│   -> ATen dispatcher: which device? which kernel?           │
│   -> decides grid size (#blocks) & block size (#threads)    │
│   -> launches kernel ASYNC, CPU moves on immediately         │
└───────────────────────────┬──────────────────────────────────┘
                            │ kernel launch = a "grid" of blocks
                            ▼
STAGE 2 — GPU (device), a chip full of SMs
┌────────────────────────────────────────────────────────────┐
│   Grid: [Block0][Block1][Block2]...[Block999]  (queue)      │
│                                                               │
│   For each block, GPU checks: does it fit an SM?             │
│     - threads <= 1024?           \                           │
│     - registers needed <= SM's?   } if NO -> launch FAILS    │
│     - shared mem needed <= SM's? /     (not a queue, an error)│
│                                                               │
│   If YES but SM is currently busy -> block WAITS in queue     │
│   until a free/matching SM slot opens up                      │
└───────────────────────────┬──────────────────────────────────┘
                            │ block assigned -> stays on 1 SM for life
                            ▼
STAGE 3 — INSIDE ONE SM
┌────────────────────────────────────────────────────────────┐
│  Resident blocks (as many as fit together):                 │
│                                                               │
│   Block A: [W0][W1][W2][W3]   Block B: [W0][W1][W2][W3]      │
│   (each W = one warp = 32 threads, split automatically)      │
│   (last warp of a block may have idle/masked lanes if the    │
│    block size wasn't a multiple of 32)                       │
│                                                               │
│  WARP SCHEDULER — every clock cycle:                         │
│   -> scan all resident warps (from A, B, ...)                │
│   -> pick one that's READY, issue its next instruction        │
│   -> a warp stalled on slow memory access is SKIPPED,         │
│      scheduler instantly switches to another ready warp       │
│      (zero-cost switch -> this is "latency hiding")           │
└───────────────────────────┬──────────────────────────────────┘
                            │ one instruction -> one chosen warp
                            ▼
STAGE 4 — ONE WARP (32 threads, lockstep / SIMT)
┌────────────────────────────────────────────────────────────┐
│  T0  T1  T2  T3  ...  T31                                   │
│  all execute the SAME instruction this cycle                 │
│  each on its OWN data (its own index into a/b)                │
│  can read/write the SM's shared memory if the kernel uses it │
└────────────────────────────────────────────────────────────┘
```

### Some Questions

#### Question

Finish by answering these without notes:

1. Starting from `torch.add(a, b)`, describe the path from the CPU call to GPU execution.
2. What is an SM?
3. What is the difference between a block and a warp?
4. How many warps are in a 256-thread block on NVIDIA hardware?
5. Why can an elementwise operation be memory-bound even though the GPU has many arithmetic units?
6. Why can several tiny kernels be slower than one fused kernel?
7. Why can a GPU show low utilization even when its kernel is fast?
8. In tiled matrix multiplication, what data is reused and where is it temporarily stored?

You are done only when answers 1–7 can be explained aloud in plain language and the answer to question 4 can be derived rather than recalled. Question 8 may remain partially uncertain at this point, but the explanation should identify shared-memory reuse. The session's architectural goal is reasoning about computation, data movement, launch overhead, and input preparation—not writing an optimized CUDA kernel.[^1]

#### Answer check

Use this only after attempting the checkpoint:

1. Python runs on the CPU; PyTorch dispatches a CUDA operation; the CPU enqueues a kernel; the GPU schedules its blocks onto SMs; each SM divides blocks into warps and issues their instructions.
2. An SM is a GPU execution cluster containing scheduling, arithmetic, register, and shared-memory resources.
3. A block is a programmer-defined cooperating group assigned to one SM; a warp is the hardware scheduling group formed from threads in that block.
4. `ceil(256 / 32) = 8` warps.
5. Each element may require only a small amount of arithmetic but still has to be read and written, so data transfer can finish more slowly than the arithmetic capacity can be used.
6. Fusion removes some launches and avoids writing and rereading intermediate tensors.
7. The GPU can finish quickly and then wait for CPU preprocessing, data transfer, synchronization, or the next request.
8. Tiles from the input matrices are loaded into shared memory and reused for multiple multiply-accumulate operations before the next tiles are loaded.

A warp is the NVIDIA scheduling/execution grouping of 32 threads, and divergent paths within a warp can require masking inactive threads.[^6][^3]


## Session 6

1. **What does autocast decide?**

   Autocast chooses the execution dtype for eligible operations during the forward pass and loss calculation. Depending on the operation, it may use FP16/BF16 for performance or FP32 for numerical stability. It does not simply convert the entire model to a lower precision. [docs.pytorch](https://docs.pytorch.org/docs/2.13/amp.html)

2. **What does autocast not decide?**

   Autocast does not scale gradients, unscale them, or adjust the gradient-scaling factor. These are separate responsibilities handled by `GradScaler` when needed. Autocast alone does not guarantee protection against underflow or overflow. [docs.pytorch](https://docs.pytorch.org/docs/2.13/amp.html)

3. **Why should backward happen outside the autocast context?**

   Autocast selects operation dtypes during the forward pass. The corresponding backward operations already use those dtype choices, so backward does not need its own autocast context. PyTorch recommends running backward outside autocast. Leaving the context does not force all backward computations into FP32; FP16 backward computations may still need gradient scaling. [docs.pytorch](https://docs.pytorch.org/docs/stable/amp.html)

4. **Why does FP16 commonly need gradient scaling?**

   FP16 has a limited range for representing small magnitudes. During backward, small nonzero gradients can underflow to zero, losing useful learning signals. Gradient scaling multiplies the loss before backward, which increases gradient magnitudes and helps them remain representable. The gradients are unscaled before the optimizer updates the parameters. [docs.pytorch](https://docs.pytorch.org/docs/stable/amp.html)

5. **Why does BF16 normally need less protection against underflow?**

   BF16 has an exponent range similar to FP32, allowing it to represent much smaller magnitudes than FP16. Small gradients are therefore much less likely to underflow, so BF16 normally does not require gradient scaling. BF16 has fewer precision bits, which means more rounding of nearby values, but rounding and underflow are different problems. Underflow is still possible—it is not eliminated. [Month2-revision1](Month2-revision1.md)

6. **Why does gradient scaling not intentionally change the final parameter update?**

   Multiplying the loss by a constant scale \(S\) also multiplies its gradients by \(S\). Before the optimizer step, those gradients are divided by the same scale, recovering the original gradients mathematically:
   $$ \frac{\partial(SL)}{\partial\theta} = S\frac{\partial L}{\partial\theta}, \qquad \frac{S\frac{\partial L}{\partial\theta}}{S} = \frac{\partial L}{\partial\theta}. $$


   The purpose is to protect small gradients during backward, not to change the effective learning rate or intended update. Floating-point rounding means the results need not be bit-for-bit identical. [docs.pytorch](https://docs.pytorch.org/docs/stable/amp.html)

### Code notes: [2_amp.ipynb](../Month%202/2_amp.ipynb)

Setup used across experiments: `batch_size=512`, `in_size=out_size=4096`, `num_layers=3`, a 3-batch-of-50 synthetic regression task (`MSELoss`, `SGD(lr=0.001)`), timed with a `start_timer()`/`end_timer()` helper that syncs CUDA and tracks `torch.cuda.max_memory_allocated()`.

1. **Baseline: default FP32 on CPU**

   ```python
   opt = torch.optim.SGD(params=model.parameters(), lr=0.001)
   start_timer()
   for epoch in range(epochs):
       for tx, ty in zip(x, y):
           logits = model(tx)
           loss = loss_fn(logits, ty)
           loss.backward()
           opt.step()
           opt.zero_grad(set_to_none=True)
   end_timer("Default precision in CPU")
   ```

   Output: `Total execution time 3.514s` for 5 epochs — the reference point with no autocast, no scaling.

2. **Autocast FP16 on CPU, single batch, isolating `backward()`**

   ```python
   with torch.autocast(device_type=device, dtype=torch.float16):
       logits = model(tx)
   with torch.autocast(device_type=device, dtype=torch.float16):
       loss = loss_fn(logits, ty)
   loss.backward()  # outside autocast, per PyTorch guidance
   ```

   Output: `total-time: 0.001724s` for one `backward()` call. This isolates that backward itself is cheap here; the autocast context only wraps the forward/loss computation, confirming point 3 above (backward runs outside autocast).

3. **Autocast FP16 on CPU, full training loop (no GradScaler)**

   ```python
   for epoch in range(epochs):
       for tx, ty in zip(x, y):
           with torch.autocast(device_type=device, dtype=torch.float16):
               logits = model(tx)
               loss = loss_fn(logits, ty)
           loss.backward()
           opt.step()
           opt.zero_grad(set_to_none=True)
   end_timer("With autograd to fp16 for 5 epochs in CPU")
   ```

   Output: `Total execution time 3.306s` for 5 epochs — only marginally faster than the FP32 baseline (3.514s), since CPU FP16 kernels aren't well optimized for this op mix and no scaler is protecting small gradients from underflow.

4. **Autocast FP16 + `GradScaler` on CUDA (1 epoch)**

   ```python
   grad_scaler = torch.amp.GradScaler(device=device)
   use_amp = True
   for epoch in range(1):
       for tx, ty in zip(x, y):
           with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
               logits = model(tx)
               loss = loss_fn(logits, ty)
           scaled_loss = grad_scaler.scale(loss)
           scaled_loss.backward()
           torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
           grad_scaler.step(opt)   # unscales grads internally, then calls opt.step()
           grad_scaler.update()
           opt.zero_grad(set_to_none=True)
   end_timer("With autograd to fp16 and gradscaler in the device cuda")
   ```

   Output: `Total execution time 0.602s` (1 epoch) and `Max memory used by tensors = 1561.42 MB` — demonstrates the production AMP pattern: `grad_scaler.scale(loss)` scales before `backward()`, `grad_scaler.step(opt)` unscales and skips the step on inf/NaN grads, and `grad_scaler.update()` adapts the scale factor for the next iteration.

**Takeaway from the runs:** FP16 autocast alone on CPU barely helps (3.514s → 3.306s) because the real speedup comes from GPU tensor cores, not just lower precision; the GPU + autocast + GradScaler combination is where the loop time drops sharply (0.602s/epoch), matching the Q&A point that autocast picks dtypes while GradScaler protects gradient magnitudes — two independent mechanisms that are typically used together on CUDA.

## Session 7

### Part A — Profiler recipe, worked through [3_profiler.py](../Month%202/3_profiler.py)

1. **Naive timing vs. profiler timing**

   ```python
   start_time = time.perf_counter_ns()
   logits = model(random_input)
   end_time = time.perf_counter_ns()
   total_time = (end_time-start_time)/(1000*1000)
   print(f"Total time taken checked naively: {total_time} milli seconds")
   ```

   A plain `perf_counter_ns()` wrap gives one wall-clock number for the whole forward pass. It cannot tell CPU time from CUDA time, cannot break the call into operators, and (on CUDA) can be wrong anyway since kernel launches are async — this is exactly the gap the Profiler is meant to fill.

2. **Basic CPU(+CUDA) profiling of one forward pass**

   ```python
   activities = [ProfilerActivity.CPU]
   if torch.cuda.is_available():
       activities += [ProfilerActivity.CUDA]
   with profile(activities=activities, profile_memory=True, record_shapes=True) as prof:
       with record_function("inference"):
           logits = model(random_input)
   prof.export_chrome_trace("trace.json")
   ```

   - `ProfilerActivity.CPU` / `.CUDA` select which activity streams get recorded; CUDA is only added if `torch.cuda.is_available()`.
   - `record_function("inference")` wraps the region with a named marker so it shows up as its own row in the table and as a labeled span in the trace, instead of everything being anonymous `aten::*` calls.
   - `profile_memory=True` adds CPU/CUDA memory columns (self and total) to the averages table.
   - `record_shapes=True` records the input tensor shapes for every op, which is what lets `group_by_input_shape=True` later split the same op into separate rows per shape.
   - `export_chrome_trace("trace.json")` writes a Chrome/Perfetto-format JSON trace that can be opened in `chrome://tracing` or [Perfetto UI](https://ui.perfetto.dev/) to see the actual timeline (launches, gaps, overlap) rather than just aggregated numbers.

3. **Reading the key-averages table**

   ```python
   print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
   print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
   print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=20))
   ```

   - `key_averages()` collapses all recorded events into one row per operator name, averaging over repeated calls.
   - Sorting by `self_cpu_time_total` surfaces the operators that are themselves expensive (excluding time spent in their children), vs. `cpu_time_total` which includes time spent in nested/child ops — this is the self time vs. total time distinction from the recipe.
   - Sorting by `self_cpu_memory_usage` surfaces the biggest allocators instead of the slowest ops — same table, different lens.
   - `help(prof.key_averages())` and the `get_sort_by_keys(event)` helper (filters `dir(event)` for names containing `time`, `memory`, or `count`) are just ways to discover which columns/attributes a `FunctionEventAvg` exposes, so you know what's valid to sort or inspect.

4. **Splitting the same operator by input shape**

   ```python
   print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total", row_limit=20))
   ```

   With `record_shapes=True` captured earlier, `group_by_input_shape=True` breaks e.g. `aten::conv2d` into multiple rows — one per distinct shape it was called with — instead of averaging all calls together. Useful for spotting whether a few unusual shapes (padding, last batch, etc.) are dominating the cost.

5. **Profiling on CUDA (model + input moved to GPU inside the `profile` block)**

   ```python
   with profile(activities=activities, profile_memory=True, record_shapes=True) as prof:
       model = resnet18().to('cuda')
       random_input = random_input.to('cuda')
       with record_function("inference"):
           logits = model(random_input)
   prof.export_chrome_trace("cuda_trace.json")
   ```

   Once tensors live on CUDA, the same `activities` list (now including `ProfilerActivity.CUDA`) lets the table show both the CPU-side launch/dispatch time *and* the actual CUDA kernel time per operator — this is the "PyTorch operator vs. CUDA kernel" and "CPU time vs. CUDA time" split the recipe asks about: a CPU row like `aten::conv2d` has a small CPU self-time (just launching) but the real work shows up as CUDA kernel time nested underneath it.

**Takeaway (Part A):** the recipe's core lesson is that a single wall-clock number can't tell you *where* time goes, while `profile()` + `record_function` + `key_averages().table(...)` can separate self vs. total time and CPU vs. CUDA time per operator, and `record_shapes=True` lets you slice that by input shape.

### Part B — Instrument the trainer, worked through [3_profiler.py](../Month%202/3_profiler.py)

```python
def build_data(batch_size, in_size, out_size, num_batches):
    x = [torch.randn(size=(batch_size, in_size)) for _ in range(num_batches)]
    y = [torch.randn(size=(batch_size, out_size)) for _ in range(num_batches)]
    return x, y

def build_model(in_size, out_size, num_layers):
    layers = []
    for _ in range(num_layers-1):
        layers.append(torch.nn.Linear(in_size, in_size))
        layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Linear(in_size, out_size))
    return torch.nn.Sequential(*tuple(layers))

batch_size, in_size, out_size, num_batches, num_layers, epochs = 512, 4096, 4096, 1, 3, 5

with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:

    with record_function("pre_training_work"):
        x,y = build_data(batch_size, in_size, out_size, num_batches)
        model = build_model(in_size, out_size, num_layers)
        loss_fn = torch.nn.MSELoss()
        opt = torch.optim.SGD(params=model.parameters(), lr=0.001)

    with record_function("training_loop"):
        for epoch in range(2): #each epoch
            with record_function("one_training_loop"):
                for tx,ty in zip(x,y): #each batch
                    with record_function("forward_pass"):
                        logits = model(tx)
                        loss = loss_fn(logits, ty)
                    with record_function("backward_pass"):
                        loss.backward()
                    with record_function("update_grad"):
                        opt.step()
                    with record_function("zero_grad"):
                        opt.zero_grad(set_to_none=True)

prof.export_chrome_trace("training_loop.json")
```

- This is the curriculum's "named regions" idea applied to a synthetic MLP instead of the CIFAR trainer: every meaningful phase of a step gets its own `record_function` label, so the exported trace and the averages table can attribute time to a *phase* (`forward_pass`, `backward_pass`, `update_grad`, `zero_grad`) instead of only to raw operator names.
- Regions nest: `pre_training_work` and `training_loop` are top-level, `one_training_loop` is nested per epoch inside `training_loop`, and `forward_pass`/`backward_pass`/`update_grad`/`zero_grad` are nested per batch inside `one_training_loop` — the trace viewer shows this as a call-stack-like hierarchy, which is what makes "self time vs. total time" meaningful at the region level too, not just per-operator.
- Because this data is synthetic tensors already sitting in Python lists (no `Dataset`/`DataLoader`, no `.to(device)`), there's no separate `data_loading` or `host_to_device` region here the way the curriculum template has — `pre_training_work` covers the equivalent "getting data and model ready" step once, outside the timed loop, rather than per-batch.
- Only `ProfilerActivity.CPU` is recorded here (CPU-only MLP), so this run isolates each phase's CPU cost; adding `ProfilerActivity.CUDA` (as in Part A step 5) would be the way to see forward/backward kernel time once this moves to GPU tensors.
- `prof.export_chrome_trace("training_loop.json")` produces one trace for the whole instrumented run (2 epochs, 1 batch each), viewable in Perfetto/`chrome://tracing` to confirm each named region lines up with the expected phase and ordering (`forward_pass` → `backward_pass` → `update_grad` → `zero_grad`, repeated per batch, repeated per epoch).

### Part C — Scheduled profiling for a short steady-state window, worked through [3_profiler.py](../Month%202/3_profiler.py)

```python
scheduler = schedule(skip_first=1, wait=1, warmup=1, active=1, repeat=4)

def trace_handler(prof):
    step = prof.step_num
    data = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10)
    print(data)
    prof.export_chrome_trace("./tmp/trace_{}.json".format(step))

with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True,
             schedule=scheduler, on_trace_ready=trace_handler) as prof:
    for epoch in range(10):
        logits = model(random_input)
        prof.step()
```

- Profiling every step of a long run is wasteful and distorts timings, so `schedule()` defines a repeating cycle: `skip_first` steps are ignored entirely, then `wait` steps are skipped (but counted), then `warmup` steps run but their data is discarded (JIT/cache warm-up), then `active` steps are actually recorded — and the whole `wait → warmup → active` cycle repeats `repeat` times.
- `prof.step()` is what advances the scheduler's internal counter; skipping happens based on this call count, not based on how many internal ops ran inside the step.
- With `skip_first=1, wait=1, warmup=1, active=1, repeat=4` over 10 steps, the active (recorded) steps land on steps 4, 7, and 10 — matching the comment in the file (`# we'll get output for 4, 7, 10`).
- `on_trace_ready=trace_handler` fires once per completed `active` window, so each one gets its own printed table and its own exported trace file (`trace_4.json`, `trace_7.json`, `trace_10.json`), letting you inspect steady-state behavior instead of a single run average contaminated by startup cost — this is the mechanism behind the curriculum's `smoke_test.json` capture (`wait=1, warmup=1, active=3, repeat=1`), just with different schedule numbers.
- `record_shapes=True` is kept on even in the scheduled run, so the recorded `active` windows still carry per-shape breakdowns, same as a one-shot profile.

#### Assignment M2.4 — answers

1. **What is the difference between CPU time and CUDA time?**
   CPU time is how long the host spends on an operator — mostly dispatch/launch overhead for a GPU op, or the actual computation for a CPU-only op. CUDA time is how long the corresponding kernel(s) actually ran on the GPU, recorded asynchronously and attributed back to the launching operator. A CPU row can show tiny CPU time but large nested CUDA time underneath it (e.g. `aten::conv2d` launching quickly, the convolution kernel itself taking much longer on-device).

2. **What is the difference between self time and total time?**
   Total time for an operator/region includes everything that happened inside it, including nested child calls. Self time subtracts out the time attributed to children, leaving only the time spent in that operator itself. `training_loop`'s total time includes `one_training_loop`, `forward_pass`, `backward_pass`, etc.; its self time would be whatever small amount of work it does that isn't inside any nested region.

3. **Why should the final benchmark run without the profiler enabled?**
   The profiler itself adds instrumentation overhead (recording events, building tables, memory tracking) that inflates step/epoch time and can also perturb things like kernel fusion or scheduling. A clean benchmark number (e.g. via CUDA events) reflects real training performance, while a profiled number reflects training-performance-plus-profiler-tax — useful for narrating *where* time goes, not for reporting *how fast* the model trains.

4. **Why should only a few steady-state steps be traced?**
   The first steps of a run include one-time costs — CUDA context/kernel compilation, cudnn autotuning, cache warm-up, allocator growth — that don't represent normal per-step cost. `schedule()`'s `wait`/`warmup`/`active` pattern (as in Part C) deliberately skips/discards those early steps so the `active` window reflects repeatable steady-state behavior, and tracing only a few steps also keeps the exported trace small enough to actually open and read.

5. **What does `record_shapes=True` add?**
   It records the input tensor shapes for every profiled operator call, which (a) shows up as a `Input Shapes` column in the averages table and (b) is what makes `key_averages(group_by_input_shape=True)` possible — splitting one operator into separate rows per distinct shape it was called with, so you can see whether a particular shape (e.g. a smaller last batch) is disproportionately expensive.

**Takeaway (Parts B & C):** named `record_function` regions turn a flat operator list into a labeled, nested timeline that maps onto the actual phases of a training step, while `schedule` + `on_trace_ready` make it practical to capture that labeled timeline for only a short, steady-state window of a long run instead of paying profiler overhead (and generating an unreadably large trace) for the entire thing.

## Session 8

### FP32 baseline, worked through [4_cifar_fp32_baseline.py](../Month%202/4_cifar_fp32_baseline.py)

- Model: the curriculum's four-conv CNN (`cifar_model`) — conv_block1 (3→64) → conv_block2 (64→64, pool) → conv_block3 (64→128) → conv_block4 (128→128, pool) → adaptive avg pool → `Linear(128, 10)`. Each conv block is `Conv2d → BatchNorm2d → ReLU`, with `MaxPool2d` added to blocks 2 and 4.
- Data: CIFAR-10 via `torchvision.datasets.CIFAR10`, transformed with `v2.Compose` — `ToImage → RandomHorizontalFlip → RandomCrop(32, padding=4) → ToDtype(float32, scale=True) → Normalize(mean=0.5, std=0.5)`. Test set uses the same deterministic tail (no random crop/flip is applied since those transforms are only meaningfully randomized on the train loader call).
- Training: `Adam(lr=0.001)`, `CrossEntropyLoss`, `batch_size=1024`, `epochs_to_run_now=15`, checkpointed every epoch via `store_checkpoint`/`load_checkpoint` (`cifarfp32_1.pth`).
- Timing: `train_one_loop` uses `torch.cuda.Event` pairs on GPU (`start.record()`/`end.record()` + `torch.cuda.synchronize()`) and `time.perf_counter_ns()` on CPU — consistent with the "don't trust an unsynchronized Python timer on CUDA" guidance from Session 7/M2.5.
- Test accuracy: computed once after training by reloading the saved checkpoint, running `model.eval()` + `torch.no_grad()`, and comparing `argmax(logits)` against labels over the full test loader.

**Device-aware timing (CUDA events vs. CPU perf counter):**

```python
if device == 'cuda':
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
else:
    start = time.perf_counter_ns()

# ... run all batches for one epoch ...

if device == 'cuda':
    end.record()
    torch.cuda.synchronize()       # wait for async kernels before reading elapsed time
    elapsed_ms = start.elapsed_time(end)
else:
    end = time.perf_counter_ns()
    elapsed_ms = (end - start) / (1000**2)
```

**Checkpoint save/load:**

```python
def store_checkpoint(model, optim, epochs, file):
    check_point = {'model': model.state_dict(), 'optim': optim.state_dict(), 'epochs_done': epochs}
    torch.save(check_point, file)

def load_checkpoint(file, device, model, optim):
    check_point = torch.load(file, map_location=device, weights_only=True)
    model.load_state_dict(check_point['model'])
    if optim is not None:
        optim.load_state_dict(check_point['optim'])
    return check_point['model'], check_point['optim'], check_point['epochs_done']
```

**Test accuracy (no grad, eval mode):**

```python
model.eval()
correct = 0
with torch.no_grad():
    for tx, ty in test_loader:
        tx, ty = tx.to(device), ty.to(device)
        logits = model(tx)
        ypred = torch.argmax(logits, dim=1)
        correct += torch.eq(ypred, ty).int().sum().item()
accuracy = correct / test_n
```

### Benchmark results (single training loop, all batches unless noted)

Note: The initial state of the CPU & GPU Models are different, which may affect the accuracy comparison.

| Run | Scope | Time | Accuracy |
|---|---|---:|---:|
| CPU alone | all batches (15 epochs) | 87 s | 70% |
| GPU + Profiler | all batches | 47 s | — |
| GPU alone | single batch | 700 ms | — |
| GPU alone | all batches (15 epochs) | 31 s (×2 runs) | 58% |

### Observations

- GPU alone (31s) is ~2.8x faster than CPU alone (87s) for the full run, as expected.
- Profiler overhead is substantial here: GPU+Profiler (47s) is ~52% slower than GPU alone (31s) for the same scope — consistent with Session 7's note that profiled time is not clean benchmark time.
- The CPU run reached higher accuracy (70%) than the GPU run (58%, reproduced twice), despite identical hyperparameters and epoch count. Since the model, optimizer, and data pipeline are unchanged across devices, this gap is more likely explained by run-to-run randomness (shuffling, augmentation, weight init) than by the device itself — this is exactly the "accidental `torch.no_grad()`" / "dataset split" style checklist from Part C worth re-checking if the gap persists after a few more seeded runs, rather than assuming GPU execution itself caused the drop.
- A single GPU batch (700 ms) includes one-time CUDA context/kernel warm-up, so it isn't a reliable per-batch estimate for the full 15-epoch run — matches the Session 5/7 point about excluding warm-up steps from steady-state timing.

## Session 9

### AMP run, worked through [5_cifar_amp.py](../Month%202/5_cifar_amp.py)

Same model, data pipeline, optimizer (`Adam(lr=0.001)`), `batch_size=1024`, and checkpointing scheme as the Session 8 FP32 baseline. The only change is the precision path: forward + loss run inside `autocast(device_type='cuda', dtype=torch.float16, enabled=True)`, and `GradScaler` scales the loss before `backward()` and unscales before `optim.step()`:

```python
with autocast(device_type=device, dtype=torch.float16, enabled=True):
    logits = model(x)
    batch_loss = loss_fn(logits, y)

scaler.scale(batch_loss).backward()
scaler.step(optim)
scaler.update()
optim.zero_grad(set_to_none=True)
```

Confirmed under autocast: `logits.dtype` is `torch.float16` — the forward/loss computation runs in FP16 while the model's stored parameters and the optimizer state remain FP32, matching the Session 6 distinction between operation dtype and parameter dtype.

### Benchmark results

| Metric | FP32 (Session 8) | AMP (Session 9) |
|---|---:|---:|
| Single-batch time | 700 ms (includes warm-up) | 620–920 ms |
| Epoch time | 31 s (15-epoch run total, see Session 8 note) | 31 s |
| Test accuracy | 58% | 68% |

Note: the FP32 "31 s" figure in Session 8 was logged as the scope "all batches (15 epochs)", while this AMP run's 31 s is logged per epoch. Scope isn't confirmed to match between the two numbers, so the epoch-time row shouldn't be read as "AMP and FP32 took the same time" without re-running both under an identical, explicitly per-epoch measurement — flagging this the same way the accuracy gap was flagged in Session 8 rather than asserting a 1.0x speedup.

### Paired-trace questions and answers

1. **Where does one training step begin and end?**

   One step begins when a batch is pulled from the `DataLoader` (`for batch_no, (x, y) in enumerate(data_loader)`) and ends after `optim.zero_grad(set_to_none=True)` completes for that batch — i.e. host-to-device transfer → forward/loss (inside autocast) → `scaler.scale(loss).backward()` → `scaler.step(optim)` → `scaler.update()` → `zero_grad`.

2. **How much time appears in host-to-device transfer?**

   About 3 ms total for moving one batch's `x` and `y` to the GPU (`x.to(device)`, `y.to(device)`).

3. **How much time appears in forward and loss?**

   Forward pass: ~3.14 ms. Loss computation (`loss_fn`) itself: ~147.2 microseconds — small, as expected for `CrossEntropyLoss` on a `[1024, 10]` logits tensor. However, calling `.item()` on the loss (to append it to `tlosses` for logging) took ~18 ms — this forces a CPU/GPU sync that stalls the async pipeline, which is far more expensive than the loss computation it's reading. This is a concrete example of accidentally introducing a sync point inside the timed region, something Session 7's "unsynchronized Python timer" caution generalizes to any blocking `.item()`/`.cpu()` call inside a hot loop.

4. **How much time appears in backward?**

   ~3.49 ms for `scaler.scale(batch_loss).backward()`.

5. **How much time appears in the optimizer?**

   ~3.027 ms for `scaler.step(optim)` + `scaler.update()`.

6. **Which stage changed most under AMP?**

   Measured with [6_fp32_vs_amp.py](../Month%202/6_fp32_vs_amp.py), input batch `10000x3x32x32` (note: larger batch than the `1024`-batch trace used in Q1–5, so absolute ms values aren't directly comparable across the two tables — only the FP32-vs-AMP delta per stage is): Below is all the CPU submitting work to the GPU for each stage.
    
   | Stage | FP32 | AMP | Notes |
   |---|---:|---:|---|
   | `forward_pass` | 5.682 ms | 5.635 ms | ~same |
   | `calculate_loss` | 0.388 ms | 0.379 ms | ~same |
   | `backward` | 3.583 ms | 4.039 ms | slightly slower under AMP (GradScaler-based `backward`) |
   | `update_gradients` | 2.709 ms | 585.063 ms | dominated by a `.item()` call costing 581.520 ms — a forced CUDA sync, not a real AMP cost |
   | `zero_grad` | 0.085 ms | 0.101 ms | ~same |


    | (CUDA) Stage | FP32 | FP16 |
    |---|---:|---:|
    | `forward_pass` time | 372 ms | 215 ms |

   Dominant convolution kernel:

   | Precision | Kernel | Time |
   |---|---|---:|
   | FP32 | `_5x_cudnn_ampere_scudnn_128x64_relu_xregs_large_nn_v1` | 11.330 ms |
   | AMP/FP16 | `_5x_cudnn_ampere_fp16_scudnn_fp16_128x64_relu_medium_nn_v1` | 5.702 ms |

   Once the `.item()` sync in `update_gradients` is discounted (same pitfall as the `batch_loss.item()` cost in Q3), the stage that genuinely changes under AMP is the convolution kernel itself: cuDNN swaps to an FP16-specialized kernel and roughly halves its time (11.330 ms → 5.702 ms), while `forward_pass`/`calculate_loss`/`zero_grad` stay effectively unchanged and `backward` is marginally slower due to `GradScaler` bookkeeping.

7. **Are visible CPU or input gaps present?**

   Yes — visible CPU-side gaps are present in the trace (e.g. around the `.item()` sync point noted in Q3), consistent with the GPU finishing its queued kernels and then waiting on host-side Python work before the next batch's kernels can be launched.

8. **Did CUDA kernel durations or names change?**

   Yes. The dominant convolution kernel differs by precision:
   - FP32: `_5x_cudnn_ampere_scudnn_128x32_relu_small_nn_v1` or `_5x_cudnn_ampere_scudnn_128x64_relu_xregs_large_nn_v1`
   - AMP/FP16: `void implicit_convolve_sgemm<__half, __half, int, 1024, 5, 5, 3, 3, 3, 1, false, false, true>(int, int, int, __half const*, int, __half*, __half const*, kernel_conv_params, unsigned long long, int, float, float, int, __half const*, __half const*, bool, bool, int, int)` or `_5x_cudnn_ampere_fp16_scudnn_fp16_128x64_relu_medium_nn_v1`

   cuDNN selects a separate, FP16-specialized kernel (note the `fp16` name segment and the `medium` vs `large` tile/register configuration) rather than running the same kernel at a different dtype — direct evidence that autocast is changing which kernel cuDNN picks, not just reinterpreting the same kernel's bits.

9. **Is the profiler observation consistent with the clean benchmark?**

   Yes — the per-region trace numbers (host-to-device, forward, backward, optimizer all in the low single-digit milliseconds, plus the `.item()` sync cost) are consistent with, and help explain, the single-batch wall-clock range of 620–920 ms once DataLoader fetch/augmentation and any remaining warm-up effects are accounted for in the clean benchmark.

### Observations

- The output logits under autocast are `torch.float16`, confirming autocast is actually taking effect for the forward pass, while the checkpointed model/optimizer state stays FP32.
- AMP reached 68% test accuracy vs. FP32's 58% from Session 8 — but per the Session 8 note, the two runs didn't share identical seeded initial weights, so this accuracy gap is more likely run-to-run variance than a precision effect, and shouldn't be read as "AMP trains to higher accuracy."
- The biggest avoidable cost found in this trace isn't precision-related at all: `batch_loss.item()` (~18 ms) dwarfs the actual loss computation (~147.2 µs) because it forces a CUDA sync every batch — worth removing from the hot loop (e.g. accumulate on-GPU and `.item()` once per epoch) before trusting any further timing comparisons.
- The clearest AMP-driven change is the cuDNN kernel swap for convolution (Q6/Q8), which is the expected mechanism for the speedup: it is not that every kernel got proportionally faster, but that a specific, lower-precision-eligible kernel class (conv/BatchNorm) got replaced, while transfer, backward, and optimizer stages stayed roughly the same order of magnitude.


## Session 10

### Final report — FP32 vs. mixed-precision CIFAR-10 training

#### 1. Environment

| Field | Value |
|---|---|
| GPU | NVIDIA L4 (compute capability (8, 9)) |
| PyTorch | 2.11.0+cu128 |
| CUDA build | 12.8 |
| cuDNN | 91900 |
| BF16 natively supported | Yes — `torch.cuda.is_bf16_supported(including_emulation=False)` returns `True` |
| Actual AMP dtype used | `torch.float16` |
| `GradScaler` used | Yes, for the FP16 path (`torch.amp.GradScaler(device, enabled=True)`) |

The Session 8/9 FP32-vs-AMP pair uses FP16 + `GradScaler`, not BF16, even though the GPU natively supports BF16. A separate precision-mode experiment ([output/cuda_fp16.txt](../Month%202/output/cuda_fp16.txt), and the CUDA BF16 run saved under the misleadingly named [output/mps_bf16.txt](../Month%202/output/mps_bf16.txt)) confirms BF16 runs on this GPU with no `GradScaler` and a near-identical accuracy curve to FP32/FP16, but that run uses a different, smaller LeNet-style model and was not profiled, so it is not used for the benchmark numbers below.

#### 2. Model and data pipeline

- Model: 4-conv CNN defined in [4_cifar_fp32_baseline.py](../Month%202/4_cifar_fp32_baseline.py) / [5_cifar_amp.py](../Month%202/5_cifar_amp.py) — `(Conv2d 3→64, BN, ReLU)` → `(Conv2d 64→64, BN, ReLU, MaxPool)` → `(Conv2d 64→128, BN, ReLU)` → `(Conv2d 128→128, BN, ReLU, MaxPool)` → `AdaptiveAvgPool2d(1,1)` → `Linear(128, 10)`.
- Parameter count: **262,218** trainable parameters (`sum(p.numel() for p in model.parameters())`, confirmed in both scripts).
- Data: `torchvision.datasets.CIFAR10`. Train transform: `ToImage → RandomHorizontalFlip → RandomCrop(32, padding=4) → ToDtype(float32, scale=True) → Normalize(0.5, 0.5, 0.5)`. Test uses the same deterministic tail without random crop/flip.

#### 3. Shared experiment configuration

| Field | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.001 (constant) |
| Scheduler | None |
| Batch size | 1024 |
| Loss function | `CrossEntropyLoss` |
| Timing method | `torch.cuda.Event` start/end pair + `torch.cuda.synchronize()` |

#### 4. FP32 result

| Field | Value |
|---|---|
| Epochs | 15 |
| Final test accuracy | 58% |
| Single-batch step time | ~700 ms (includes one-time CUDA/cuDNN warm-up) |
| 15-epoch wall time | 31 s (≈ 2.07 s/epoch average, including the warm-up epoch) |

Accuracy stayed below the 70% target hit by the earlier CPU run (70%, 87 s) with identical hyperparameters — flagged in Session 8 as more likely run-to-run variance (shuffling, augmentation, weight init) than a device effect.

#### 5. Mixed-precision result

| Field | Value |
|---|---|
| Epochs | 1 (see Limitations) |
| Final test accuracy | 68% |
| Single-batch step time | 620–920 ms |
| Logged "epoch time" | 31 s (scope not confirmed comparable to the FP32 figure — see Section 6) |

#### 6. Benchmark comparison

| Metric | FP32 | AMP (FP16) |
|---|---:|---:|
| Step time | ~700 ms | 620–920 ms |
| Test accuracy | 58% | 68% |
| Dominant conv kernel time (isolated 10k-image batch, [6_fp32_vs_amp.py](../Month%202/6_fp32_vs_amp.py)) | 11.330 ms (`..._scudnn_128x64_relu_xregs_large_nn_v1`) | 5.702 ms (`..._fp16_scudnn_fp16_128x64_relu_medium_nn_v1`) |

| Stage | FP32 | FP16 |
|---|---:|---:|
| `forward_pass` time | 372 ms | 215 ms |

Two speedup numbers come out of this data, and they disagree, which is itself the headline finding:

- **Step-time ratio (wall clock, real DataLoader batches):** 700 / 620–920 ≈ **0.76x–1.13x** — essentially no reliable end-to-end win, within run-to-run noise.
- **Kernel-level ratio (isolated conv kernel, synthetic batch):** 11.330 / 5.702 ≈ **1.99x** — close to the theoretical 2x for the one op that is actually FP16-accelerated.

The 15-epoch wall-time rows are not used for the headline ratio: the FP32 "31 s" figure covers all 15 epochs, while the AMP run only executed 1 epoch, so the two "31 s" numbers in the raw logs are coincidental, not a like-for-like per-epoch comparison (flagged already in Session 9 notes).

#### 7. Profiler observations

From the paired-trace work in Session 9 ([cifarfp32_cuda_single_batch_training_loop.json](../Month%202/cifarfp32_cuda_single_batch_training_loop.json), [cifaramp_cuda_single_batch_training_loop.json](../Month%202/cifaramp_cuda_single_batch_training_loop.json), [cuda_simple_fp32_trace.json](../Month%202/cuda_simple_fp32_trace.json), [cuda_simple_amp_trace.json](../Month%202/cuda_simple_amp_trace.json)):

- Host-to-device transfer: ~3 ms/batch, unchanged between FP32 and AMP.
- Forward pass: ~5.68 ms (FP32) vs. ~5.64 ms (AMP) — effectively unchanged at the region level despite the conv kernel itself being ~2x faster, because the region also includes BatchNorm/ReLU/pooling ops and layout-conversion kernels (`nchwToNhwcKernel`/`nhwcToNchwKernel`) that run regardless of precision.
- Backward: ~3.58 ms (FP32) vs. ~4.04 ms (AMP) — slightly *slower* under AMP due to `GradScaler` bookkeeping.
- Optimizer step: ~2.71 ms (FP32) vs. ~3.54 ms (AMP, after removing a ~581 ms `.item()`-induced CUDA sync artifact that is not a real AMP cost).
- `zero_grad`: ~0.09–0.10 ms, unchanged.
- cuDNN selects a *different, FP16-specialized kernel* for the convolution under autocast (`implicit_convolve_sgemm<__half,...>` / `_5x_cudnn_ampere_fp16_scudnn_fp16_...`) rather than reinterpreting the same kernel's bits — direct evidence that autocast changes kernel selection, not just storage format.
- A large avoidable CPU/GPU sync point exists in the hot loop (`batch_loss.item()`, ~18 ms per batch) that dwarfs the actual loss computation (~147 µs); this is a pipeline inefficiency, not a precision effect, and it would mask any real AMP speedup if left in a clean benchmark.

#### 8. Why the speedup is not 2x

Summing the per-stage numbers from Section 7 (excluding the `.item()` sync artifact) gives roughly 12.45 ms/step for FP32 and 13.70 ms/step for AMP on the synthetic 10,000-image profiled step — i.e. the one operation that halves in time (the convolution kernel, 11.33 ms → 5.70 ms) is only part of a step whose other parts stay the same size or grow slightly. This matches the curriculum's checklist point by point:

- **Autocast does not convert every operation to FP16.** Only the conv/BatchNorm path inside `autocast` swaps kernels (Section 7); `BatchNorm`'s own reduction/normalization and the loss computation run in types chosen by PyTorch's op-level allow/deny list, not uniformly in FP16.
- **Numerically sensitive ops stay in FP32.** `CrossEntropyLoss`'s softmax/log-sum-exp stays FP32-stable regardless of autocast, consistent with `calculate_loss` showing no measurable change (0.388 ms → 0.379 ms).
- **Parameters and optimizer state stay FP32.** Checkpoints saved by `store_checkpoint` contain FP32 `model.state_dict()`/`optim.state_dict()` even for the AMP run — only the forward/loss activations are FP16 (`logits.dtype == torch.float16`), so Adam's moment updates in `update_gradients` do not benefit from the lower-precision kernel and instead got slightly slower under AMP (2.71 ms → 3.54 ms) because of `GradScaler`'s unscale/step bookkeeping.
- **Data loading and CPU transforms don't speed up.** `RandomCrop`/`RandomHorizontalFlip`/`Normalize` run on the CPU in Python/`torchvision.transforms.v2` and are unaffected by the GPU's autocast context — not separately measured here, but they are a fixed, precision-independent floor under every batch.
- **Host-to-device transfer doesn't halve.** Measured transfer stayed ~3 ms either way (Session 9) because the images are read, cropped, and transferred as FP32 tensors before the model is ever reached; FP16 only exists inside the forward/loss region.
- **Kernel-launch overhead remains.** The layout-conversion kernels (`nchwToNhwcKernel`, `nhwcToNchwKernel`) appear in both FP32 and FP16 traces (Section 7 trace file inspection) — these launches aren't eliminated by autocast, they're just launched with half-size tensors.
- **The optimizer gets little benefit from Tensor Cores.** Adam's elementwise moment/update math is not a matmul/conv op, so it isn't autocast-eligible and isn't Tensor-Core-accelerated — observed directly as a slower, not faster, optimizer stage under AMP.
- **This CNN may not generate enough matrix work to saturate Tensor Cores.** With `batch_size=1024` and channel widths of 64/128, the conv workload is modest next to a GPU recipe tuned for much larger matmuls; the ~2x the convolution *does* achieve suggests the op itself is Tensor-Core-eligible, but the surrounding small ops (BatchNorm, pooling, zero_grad) never reach a scale where precision matters.
- **Halving representation size doesn't halve every byte moved in the full step.** Only the forward/loss tensors are FP16; inputs, parameters, gradients, and optimizer state (momentum/variance buffers) stay FP32, so total step memory traffic shrinks by far less than 2x.
- **End-to-end speedup is limited by the accelerated fraction of the step.** Using Amdahl's-law reasoning on the Section 7 numbers: the conv kernel is roughly 11.33/12.45 ≈ 91% of the *measured forward compute* here but a much smaller fraction of the full step once transfer, backward, optimizer, and `.item()`-style sync points are included — which is exactly why the step-time ratio (0.76x–1.13x) comes out far below the kernel-level ratio (1.99x).

#### 9. Limitations

- The AMP run only executed 1 epoch, not the full 15 used for the FP32 baseline, so the "epoch time" row in the raw logs is not a controlled comparison; only the single-batch step-time range and the isolated-kernel profiler numbers are treated as reliable here.
- Neither run reached the 70% accuracy target reached by the earlier CPU baseline; FP32 (58%) and AMP (68%) are also not confirmed to share identical seeded initial weights, so the accuracy gap between them is more likely run-to-run variance than a precision effect.
- GPU utilization was not measured with `nvidia-smi` or any other tool for either run, so that required field is not available and is reported here as not measured rather than guessed.
- The 2x-ratio comparison in Section 7 came from a single synthetic, unprofiled-pipeline batch (`6_fp32_vs_amp.py`, random 10,000×3×32×32 tensors, no DataLoader/augmentation), so it isolates stage costs well but is not itself a clean end-to-end benchmark.
- BF16 was never run through the same profiled FP32/AMP trace pair — the only BF16 evidence comes from a separate, smaller LeNet-style model/script, so this report cannot compare BF16 against FP16 on the same 4-conv CNN.
- The `batch_loss.item()` sync cost (~18 ms/batch) was identified but not removed before the benchmark numbers above were collected, so the reported AMP step times still include some of that avoidable overhead.

### Oral Report

1. Why is a GPU not simply a faster CPU?

A GPU is not simply a faster CPU because they prioritize different kinds of performance. A CPU is designed to execute individual instruction streams quickly, including work with dependencies and complex control flow. A GPU is designed for high throughput by executing many threads concurrently. This makes a GPU well suited to large amounts of independent work, but adding more execution lanes cannot remove a sequential dependency chain.

2. What does the CPU do when a CUDA kernel is launched?

When a CUDA kernel is launched, the CPU runs the host-side code needed to submit the kernel and its arguments to a CUDA stream. The launch is normally asynchronous: the CPU can continue executing and submitting more work without waiting for the GPU to finish. Operations within the same stream execute in order. Synchronization or some operations like `item()` which need the info to be transfered to the CPU, forces the CPU to wait when completion is required.

3. What is an SM?

An SM, or Streaming Multiprocessor, is a hardware unit within an NVIDIA GPU containing multiple execution units, warp schedulers, registers, and shared memory. Thread blocks are assigned to SMs, and each block’s threads are organized into warps. Multiple blocks can reside on one SM when resources permit, and the schedulers issue instructions from ready warps.

4. What is a warp?

A warp is a group of 32 threads that an SM schedules for execution. When an instruction is issued for a warp, the participating threads execute that instruction using their own data and private registers. Threads in the warp belong to the same block, but have different thread indices.

| Memory        | Who can access it?                                                    |
| ------------- | --------------------------------------------------------------------- |
| Registers     | The owning thread                                                     |
| Shared memory | Threads within the same thread block                                  |
| Global memory | Threads across different blocks and SMs, given an appropriate address |

5. Why is shared memory useful when global memory already exists?

Shared memory provides fast, programmer-managed, on-chip storage that threads within a block can share. It is useful for reusing data and exchanging intermediate results, reducing repeated global-memory accesses. That can improve throughput when memory access is limiting performance.

6. What is the difference between compute-bound and memory-bound work?

Compute-bound work is limited primarily by how quickly the processor can execute calculations. Memory-bound work is limited primarily by accessing or moving data. For a memory-bandwidth-bound kernel, increasing arithmetic capacity alone offers little benefit; increasing memory bandwidth or reducing memory traffic can help. Arithmetic intensity—operations per byte moved—helps us reason about which limit is more likely.

7. What is launch-bound work?

Launch-bound work is limited by the overhead or rate of launching kernels rather than by their arithmetic or memory throughput. It commonly occurs with many tiny kernels whose execution is too short to hide the launch overhead. I would look for short kernels and GPU idle gaps aligned with CPU-side submission activity, rather than simply adding launch-call durations to GPU time.

8. Why can GPU utilization be high while execution remains inefficient?

GPU utilization can be high because kernels remain active, but that does not show how effectively they use the hardware. They might perform redundant work or use execution resources poorly. However, low arithmetic throughput alone is not proof of inefficiency: a memory-bound kernel may already be efficient if it approaches its relevant memory-bandwidth limit.

9. What does autocast decide?

Autocast chooses execution dtypes for eligible operations using predefined, device-specific policies. It runs suitable operations, such as matrix multiplication and convolution, in lower precision for performance, while retaining FP32 for operations whose numerical behavior benefits from it. It does not permanently convert the model’s FP32 parameters, and it does not dynamically test whether each input is safe in FP16.

Eg: 

```text
total = a large value
add a very small value to total #need fp32 here for precision
repeat many times
```

10. Why does FP16 commonly need gradient scaling?

FP16 has a limited representable range, so small gradients can underflow to zero during backward. Gradient scaling multiplies the loss before backward, which scales the gradients and helps preserve those small values. The gradients are then unscaled before the optimizer updates the parameters.
pytorch

11. Why does BF16 normally have less underflow risk than FP16?

```
FP32 range:
  max           : 01111111011111111111111111111111  ->  3.402823e+38
  min (lowest)  : 11111111011111111111111111111111  ->  -3.402823e+38
  min normal    : 00000000100000000000000000000000  ->  1.175494e-38
  min subnormal : 00000000000000000000000000000001  ->  1.401298e-45

FP16 range:
  max           : 0111101111111111  ->  6.550400e+04
  min (lowest)  : 1111101111111111  ->  -6.550400e+04
  min normal    : 0000010000000000  ->  6.103516e-05
  min subnormal : 0000000000000001  ->  5.960464e-08

BF16 range:
  max           : 0111111101111111  ->  3.389531e+38
  min (lowest)  : 1111111101111111  ->  -3.389531e+38
  min normal    : 0000000010000000  ->  1.175494e-38
  min subnormal : 0000000000000001  ->  9.183550e-41
```

BF16 has 8 exponent bits, compared with FP16’s 5, so it can represent much smaller nonzero magnitudes and normally has less underflow risk. FP16 has more fraction bits—10 versus BF16’s 7—so it offers finer precision within its normal range, but a narrower range overall.

12. Why must CUDA timing account for asynchronous execution?

CUDA launches are asynchronous, so a CPU timer can finish measuring before GPU execution completes. For wall-clock timing, synchronize before starting to exclude earlier queued work and before stopping to include completion of the measured work. Alternatively, use CUDA events and wait for the ending event before reading their elapsed time.

13. Why is clean benchmark timing separated from profiler timing?

Clean timing is separated from profiling because collecting execution events, shapes, and stack traces adds overhead and may change runtime behavior. I use unprofiled, warmed-up, correctly synchronized measurements to establish performance, then profiler traces to explain bottlenecks. I verify any improvement with another clean benchmark.

14. How would a DataLoader bottleneck appear in a trace?

A DataLoader bottleneck would appear as a long batch-retrieval region on the CPU overlapping a GPU idle gap. The next batch’s transfer and computation cannot proceed until retrieval completes. A long DataLoader region alone is not sufficient evidence, because loading may overlap useful GPU work.

15. Why was your measured AMP speedup not 2x?

AMP did not give a 2× end-to-end speedup because it did not halve the duration of every part of my training step. I observed faster convolutions under FP16, but other work remained, including operations executed in FP32. AMP can also add dtype-conversion and gradient-scaling work. Overall speedup depends on how much of the original runtime was accelerated, how much those operations improved, and any additional overhead—not simply on FP16 having half as many bits as FP32.
