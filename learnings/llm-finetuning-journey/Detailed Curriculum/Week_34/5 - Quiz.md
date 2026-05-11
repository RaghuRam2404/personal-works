# Week 34 Quiz — Unsloth Optimizations

## Multiple Choice

**Q1.** Unsloth claims 2–5x faster training than vanilla HuggingFace QLoRA. Which of the following is NOT a primary optimization that Unsloth applies?

A. Fused LoRA kernel that computes A and B matrix multiplications in a single CUDA/Triton kernel  
B. Custom gradient checkpointing implementation that reduces VRAM beyond PyTorch's default  
C. Automatic model parallelism across multiple GPUs for linear speedup  
D. Custom RoPE (Rotary Position Embedding) precomputation kernel

---

**Q2.** You run vanilla QLoRA training on Qwen2.5-Coder-7B with batch size 4 and get 1.0 steps/second and 24GB peak VRAM. After switching to Unsloth with identical hyperparameters, you observe 2.3 steps/second and 14GB peak VRAM. What would you do with the freed VRAM?

A. Reduce `max_seq_length` to free even more VRAM  
B. Increase batch size from 4 to 8 (or more), improving throughput further and potentially quality  
C. Enable FP32 precision for higher quality training  
D. Load a second model to compare in parallel

---

**Q3.** When using Unsloth, you should set `use_gradient_checkpointing="unsloth"` in `FastLanguageModel.get_peft_model()`. What should you NOT simultaneously set in `SFTConfig`?

A. `packing=True`  
B. `gradient_checkpointing=True` (setting both causes a conflict)  
C. `optim="paged_adamw_8bit"`  
D. `lr_scheduler_type="cosine"`

---

**Q4.** You run the exact same training (same data, same hyperparameters, same number of steps) with vanilla QLoRA and Unsloth. At step 1000, vanilla QLoRA has eval loss 1.18 and Unsloth has eval loss 1.22. Which conclusion is most appropriate?

A. Unsloth has a bug — its kernels are numerically incorrect  
B. The difference (0.04 loss units) is within typical stochastic training variance; quality is effectively equivalent  
C. Vanilla QLoRA is clearly superior; Unsloth should not be used for quality-sensitive tasks  
D. Unsloth's dropout=0 recommendation caused underfitting

---

**Q5.** Unsloth does not support all HuggingFace models. If you wanted to fine-tune a model that Unsloth does not support (e.g., a new architecture), which library would you fall back to?

A. DeepSpeed ZeRO-3 (always the fallback for any unsupported model)  
B. Vanilla HuggingFace peft + bitsandbytes + SFTTrainer (the Week 33 setup)  
C. PyTorch FSDP (Fully Sharded Data Parallel)  
D. Axolotl (which re-implements Unsloth's kernels for all models)

---

## Short Answer

**Q6.** Flash Attention 2 is one of Unsloth's performance improvements. Explain in 3–4 sentences how Flash Attention 2 differs from standard attention in terms of memory complexity, and why this matters specifically for long sequence training (e.g., `max_seq_length=2048`).

---

**Q7.** Your Unsloth training run on Qwen2.5-Coder-7B shows 3.2 steps/second on an A100. You estimate you need to train on 15K examples for 2 epochs. Approximately how many training steps will this be (assuming packing with average 2 examples per packed sequence), and how long will training take?

---

**Q8.** A colleague proposes using Unsloth on their Mac M3 Max for local development before running full training on Colab Pro A100. Is this a good idea? What should they do instead for local testing?

---

## Scenario

**Q9.** You are planning the Week 38 domain sprint: QLoRA fine-tune Qwen2.5-Coder-7B on 15K examples, 2 epochs, with Unsloth on Colab Pro A100. You have a $10 Colab Pro credit remaining. Estimate the total training time and cost, and decide whether to proceed or optimize first.

Use these numbers: A100 Colab Pro costs approximately $1.20/hour. Unsloth achieves 3 steps/second on your setup. With packing at seq_length 512, average pack ratio is ~3 examples per sequence.
