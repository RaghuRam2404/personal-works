
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

