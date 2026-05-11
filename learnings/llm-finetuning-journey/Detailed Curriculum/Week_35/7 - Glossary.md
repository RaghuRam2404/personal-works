# Week 35 Glossary

**Learning rate (LR)**: The step size multiplied by the gradient during each parameter update; the single most impactful hyperparameter in LoRA fine-tuning.

**Effective batch size**: `per_device_batch_size × gradient_accumulation_steps × num_gpus`; the number of examples whose gradients are averaged before each optimizer step.

**Gradient accumulation**: Performing multiple forward/backward passes before calling `optimizer.step()`; effectively simulates a larger batch size.

**LR scheduler**: A function that modifies the learning rate over the course of training (e.g., cosine decay, linear decay); controls how aggressively LR decreases from peak to minimum.

**Warmup**: A period at the start of training where LR linearly increases from near-0 to the target LR; prevents instability from large gradients applied to random LoRA matrices.

**`warmup_ratio`**: Fraction of total training steps used for LR warmup; `warmup_ratio=0.05` means the first 5% of steps ramp LR from 0 to target.

**Early stopping**: Stopping training when a monitored metric (typically eval loss) stops improving; implemented via `load_best_model_at_end=True` in `SFTConfig`.

**W&B sweep**: A structured hyperparameter search using Weights & Biases, supporting grid search, random search, or Bayesian optimization over user-defined parameter spaces.

**Linear scaling rule**: A heuristic stating that optimal LR scales linearly with batch size (double batch → double LR); often too aggressive for LLMs; square root scaling is more conservative.

**Square root scaling rule**: A more conservative batch-size/LR heuristic: LR_new = LR_old × sqrt(batch_new / batch_old).

**NEFTune**: A technique that adds random noise to embeddings during SFT training; sometimes improves instruction-following quality; controlled via `neftune_noise_alpha`.

**`load_best_model_at_end`**: `SFTConfig` option that reloads the checkpoint with the best eval metric at training end; essential for early stopping to actually use the best checkpoint.
