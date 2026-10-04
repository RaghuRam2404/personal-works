
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
