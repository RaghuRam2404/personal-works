# Week 21 Glossary — Running Pretraining

**Loss spike**: A sudden upward jump in training loss during an otherwise stable training run; usually caused by a bad batch or temporarily extreme gradients.

**NaN loss**: Not-a-Number appearing in the loss value, caused by overflow, underflow, or division by zero during the forward or backward pass; terminates a training run if unhandled.

**BF16 (bfloat16)**: A 16-bit floating-point format with 8 exponent bits and 7 mantissa bits; same dynamic range as FP32, less precision than FP16; preferred for training on A100 GPUs.

**GradScaler**: A PyTorch utility that scales loss values to prevent FP16 underflow during backpropagation; not needed with BF16.

**Gradient clipping**: Rescaling the gradient vector when its L2 norm exceeds a threshold (typically 1.0); prevents parameter updates from being too large.

**MFU (Model FLOP Utilization)**: Fraction of theoretical peak FLOP/s actually utilized; reflects GPU efficiency including memory bandwidth and communication overhead.

**Warmup steps**: Initial training steps where the learning rate increases linearly from 0 to max_lr; prevents large gradient steps when optimizer states are near zero.

**Cosine annealing**: A learning rate schedule that decreases from max_lr to min_lr following a cosine curve; standard for language model pretraining.

**Checkpoint**: A saved snapshot of model weights and optimizer states at a specific training step; enables resuming training after interruption.

**Effective batch size**: The product of per-GPU batch size × number of GPUs × gradient accumulation steps; the number of samples the gradient is computed over before an optimizer update.

**Perplexity**: exp(cross_entropy_loss); measures how surprised the model is by text; a 56M model should achieve ~20–30 on held-out web text.

**Training throughput (tokens/sec)**: Number of tokens processed per second; limited by GPU memory bandwidth, compute, and data loading speed.

**Epoch**: One complete pass through the training dataset; in pretraining, multiple epochs can cause memorization if the dataset is small.
