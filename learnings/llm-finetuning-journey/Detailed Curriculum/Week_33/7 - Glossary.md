# Week 33 Glossary

**QLoRA**: Quantized LoRA — fine-tuning method combining a frozen 4-bit NF4 base model with trainable BF16 LoRA adapters; enables 7B training on a single 24GB GPU.

**Paged optimizer**: An optimizer (implemented in bitsandbytes) that allows optimizer states to be transparently paged from GPU to CPU RAM under memory pressure, preventing OOM crashes.

**paged_adamw_8bit**: bitsandbytes 8-bit AdamW optimizer with paging support; reduces optimizer state memory 4x compared to standard fp32 AdamW.

**Storage dtype**: The precision used to store model weights in GPU memory (e.g., NF4 for QLoRA base model). Distinct from compute dtype.

**Compute dtype**: The precision used during matrix multiplication after on-the-fly dequantization (BF16 in QLoRA). Set via `bnb_4bit_compute_dtype`.

**On-the-fly dequantization**: bitsandbytes behavior where NF4 weights are temporarily dequantized to BF16 for each matrix multiply, then discarded — the stored NF4 weights are never changed.

**Gradient checkpointing**: Training technique that recomputes activations during the backward pass instead of storing them, trading compute for memory (~60–70% activation memory reduction).

**`model.config.use_cache = False`**: Required setting before enabling gradient checkpointing in decoder-only models; disables the KV cache that conflicts with gradient checkpointing.

**Activation memory**: GPU memory consumed by intermediate activations during the forward pass that must be retained for the backward pass; proportional to batch size × sequence length × model width.

**`model.print_trainable_parameters()`**: peft method that prints the count and percentage of trainable parameters in a PEFT-wrapped model; useful for verifying LoRA is applied correctly.
