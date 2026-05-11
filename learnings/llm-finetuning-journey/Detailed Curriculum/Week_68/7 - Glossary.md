# Week 68 Glossary — Training Pipeline Documentation

**Effective batch size**: The product of per-device batch size, gradient accumulation steps, and number of GPUs; the actual number of examples processed before each weight update.

**Gradient clipping**: Capping the norm of gradient vectors at a maximum value (typically 1.0) to prevent exploding gradients; a required hyperparameter to document for reproducibility.

**Weight decay**: L2 regularization coefficient applied to model weights in AdamW; prevents overfitting; typical value 0.01–0.1.

**LoRA alpha (α)**: Scaling factor applied to LoRA's weight update: effective update = (α/r) × BA; convention α = 2r keeps the effective LR stable across different rank values.

**Reward margin (DPO)**: The average difference in log-probability assigned to chosen vs rejected responses after DPO training; a training diagnostic, not the primary quality metric.

**β (DPO beta)**: The KL penalty coefficient in DPO's loss; small β allows large policy updates away from the SFT reference; large β keeps the policy conservative.

**Reference policy**: The frozen SFT model used as the KL baseline in DPO; its log-probabilities appear in the denominator of the DPO loss.

**GPU-hours**: The product of number of GPUs and wall-clock hours; the standard unit for reporting training compute.

**Compute budget**: The section of a technical report documenting total GPU-hours, hardware type, and estimated cost; required for reproducibility.

**Ablation study**: A controlled experiment removing one component of a pipeline to measure its isolated contribution; cited as a table or inline number in the training section.
