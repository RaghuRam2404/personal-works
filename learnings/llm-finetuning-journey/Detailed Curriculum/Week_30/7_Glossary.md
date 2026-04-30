# Week 30 Glossary

**LoRA (Low-Rank Adaptation)**: A parameter-efficient fine-tuning method that trains two small matrices A and B such that delta_W = BA approximates the full weight update.

**Rank (r)**: The bottleneck dimension of LoRA adapters; controls the number of independent directions the adapter can express. Typical values: 8, 16, 32, 64.

**alpha**: A LoRA scaling hyperparameter; the adapter output is multiplied by alpha/r. Decouples update magnitude from rank.

**lora_A**: The "down projection" matrix in LoRA; shape (r, d_in), initialized with Kaiming uniform.

**lora_B**: The "up projection" matrix in LoRA; shape (d_out, r), initialized to zero to ensure a no-op adapter at step 0.

**LoRA scaling factor**: alpha / r — multiplies the LoRA output before adding to the pretrained output; controls the relative magnitude of the adapter contribution.

**Intrinsic dimensionality**: The effective number of parameters needed to reach near-optimal performance on a task; empirically much smaller than the full model parameter count.

**Adapter merging**: Computing W_merged = W + (B @ A) * (alpha/r) and discarding A and B; removes LoRA inference overhead.

**Target modules**: The set of linear layer names to which LoRA adapters are applied (e.g., "q_proj", "v_proj"); broader coverage generally improves performance.

**Kaiming uniform initialization**: Weight initialization scheme suitable for layers followed by ReLU/linear activations; used for lora_A to ensure non-zero gradient flow from step 1.

**Parameter efficiency**: The ratio of fine-tuning performance gain to number of trainable parameters; LoRA achieves near full-SFT performance with ~0.1–1% of parameters.

**Intrinsic low-rank hypothesis**: The empirical observation that weight change matrices (delta_W) during fine-tuning have low effective rank, justifying LoRA's approximation.
