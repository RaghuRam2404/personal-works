# Week 15 Glossary

**Mixed-precision training**: Using lower-precision floats (bfloat16) for forward pass while keeping FP32 for optimizer states; faster with near-equivalent quality.

**bfloat16 (BF16)**: 16-bit float with the same 8-bit exponent as FP32 but only 7-bit mantissa; avoids overflow; preferred over FP16 on A100/H100.

**torch.autocast**: PyTorch context manager that automatically casts operations to a lower dtype (BF16) during the forward pass.

**Gradient accumulation**: Running multiple micro-batch forward passes before a single optimizer step to simulate a larger effective batch; requires dividing loss by num_steps before backward.

**Effective batch size**: `B × T × grad_accum_steps`; the logical batch size seen by the optimizer per step.

**Gradient clipping**: Scaling down the gradient when its norm exceeds a threshold (typically 1.0); prevents instability from occasional large gradients.

**Cosine learning rate decay**: LR schedule that decreases from max_lr to min_lr following a cosine curve after a linear warmup period.

**Flash Attention**: IO-aware attention algorithm that tiles computation to avoid materializing the full T×T attention matrix in HBM; O(T) memory instead of O(T²).

**HBM (High Bandwidth Memory)**: GPU DRAM; slower than SRAM. Flash Attention avoids slow HBM reads/writes of the attention matrix.

**HellaSwag**: Commonsense NLI benchmark used to evaluate language model quality during pretraining; framed as next-sentence-prediction with 4 candidates.

**tiktoken**: OpenAI's fast tokenizer library; used for GPT-2 (50257 vocab), GPT-4, and other OpenAI models.

**Weight decay exclusion**: Applying L2 weight decay only to 2D weight matrices (not biases, embeddings, LayerNorm params); standard practice in transformer training.

**Checkpoint**: Saved model (and optimizer) state at a point during training; enables resuming or using partially-trained models.

**OpenWebText**: Open-source reproduction of the dataset used to train GPT-2; ~40GB of web text from Reddit outlinks with ≥3 karma.
