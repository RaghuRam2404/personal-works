# Week 5 — Glossary

**SGD (Stochastic Gradient Descent)**: Basic optimizer that updates parameters by subtracting the learning rate times the gradient.

**Momentum**: Addition to SGD that accumulates an exponential moving average of past gradients, dampening oscillations and accelerating convergence.

**Adam**: Adaptive optimizer maintaining per-parameter first and second gradient moment estimates with bias correction.

**AdamW**: Adam with decoupled weight decay — weight decay is applied directly to parameters, bypassing adaptive scaling.

**Weight decay**: Regularization that penalizes large parameter values by subtracting a fraction of the parameter at each step; equivalent to L2 regularization in SGD, but distinct in Adam.

**L2 regularization**: Adding `λ * ||θ||²` to the loss function, which is equivalent to weight decay in SGD but not in Adam.

**Bias correction**: Adjustment in Adam to account for the zero-initialization of moment estimates — divides by `(1 - β^t)` to avoid early underestimation.

**Learning rate (LR) schedule**: A function that changes the learning rate over training; warmup + cosine decay is the current standard.

**Linear warmup**: Gradually increasing the LR from near-zero to the target LR at the start of training; prevents early instability.

**Cosine decay**: Decreasing the LR following a cosine curve from `max_lr` to `min_lr` after warmup; produces smooth convergence.

**Gradient clipping**: Rescaling the global gradient norm to a maximum value before the optimizer step; prevents gradient explosions.

**Global gradient norm**: The L2 norm of all gradients concatenated into a single vector; used as the clipping threshold.

**AMP (Automatic Mixed Precision)**: PyTorch feature that automatically runs forward pass in FP16/BF16 while keeping master weights in FP32.

**GradScaler**: PyTorch class that dynamically scales loss before backward to prevent FP16 underflow, then unscales before optimizer step.

**FP16 (float16)**: 16-bit floating point; ~2× faster than FP32 on modern GPUs; narrower dynamic range (max ~65504).

**BF16 (bfloat16)**: 16-bit format with same exponent range as FP32 but less mantissa precision; preferred over FP16 for training on A100/H100.

**Loss scaling**: Multiplying the loss by a large factor before backward to prevent FP16 underflow in gradients; GradScaler handles this automatically.
