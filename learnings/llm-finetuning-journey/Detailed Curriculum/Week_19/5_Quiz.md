# Week 19 Quiz — Distributed Training

## Multiple Choice

**Q1.** In standard DDP (Distributed Data Parallelism), each GPU holds:

A) A shard of the model parameters  
B) A full copy of the model parameters  
C) Only the optimizer states, with parameters stored on GPU 0  
D) A pipeline stage (a subset of layers)  

---

**Q2.** ZeRO-2 partitions which components across GPUs?

A) Parameters only  
B) Gradients only  
C) Optimizer states and gradients  
D) Optimizer states, gradients, and parameters  

---

**Q3.** You are training a 7B model on 8 A100 80GB GPUs. Each GPU has 24.5GB of parameter + optimizer + gradient memory under ZeRO-2 (ignoring activations). If you switch to ZeRO-3, the per-GPU parameter memory drops from 14GB to approximately:

A) 7GB  
B) 3.5GB  
C) 1.75GB  
D) 0.87GB  

---

**Q4.** HuggingFace Accelerate's primary value for a single-GPU user (like your Colab setup) is:

A) Enabling model parallelism across CPU and GPU  
B) Providing gradient accumulation, mixed precision, and portability to multi-GPU without code changes  
C) Automatically selecting the best learning rate via adaptive algorithms  
D) Replacing the need for an optimizer by using gradient-free methods  

---

**Q5.** Which statement about gradient accumulation is correct?

A) It is mathematically equivalent to a larger batch size for transformer models (no BatchNorm)  
B) It reduces GPU memory usage by storing only partial gradients  
C) It is faster than processing a large batch in a single step  
D) It requires a distributed setup to work correctly  

---

## Short Answer

**Q6.** Explain in 3 sentences why ZeRO-3 (full sharding) can actually be slower than ZeRO-2 on some hardware configurations, despite providing better memory efficiency.

---

**Q7.** You are fine-tuning Qwen2.5-14B on 2 A100 80GB GPUs. Using the memory analysis approach from the assignment (14B params FP16, gradients FP32, Adam optimizer states FP32), compute the per-GPU memory under ZeRO-2 and determine if it fits.

---

**Q8.** Your training script runs fine on a single GPU. You want to run it on 4 GPUs using Accelerate. List the 3 changes you need to make to your script and config to enable this.

---

## Scenario

**Q9.** A colleague hands you a training script for a 13B model that uses plain PyTorch DDP. The model runs out of memory even on 4 A100 80GB GPUs.

1. What is the approximate per-GPU memory requirement under DDP? (13B params, Adam optimizer, FP16 params, FP32 gradients and optimizer states)
2. Which ZeRO stage would you recommend to make it fit on 4 A100 80GB GPUs?
3. Show the minimal Accelerate config change needed to switch from DDP to DeepSpeed ZeRO-2
4. After switching, training is 15% slower than DDP was on 2 GPUs. Is this expected? Why?
