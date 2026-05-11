# Week 31 Glossary

**peft**: HuggingFace library (Parameter-Efficient Fine-Tuning) providing implementations of LoRA, DoRA, prefix tuning, and other adapter methods.

**LoraConfig**: The peft configuration dataclass specifying rank, alpha, target_modules, dropout, bias strategy, and task type for a LoRA fine-tuning run.

**get_peft_model**: peft function that wraps a base HuggingFace model, replacing target linear layers with LoRA-augmented versions and freezing all non-adapter parameters.

**target_modules**: The list of linear layer name substrings (e.g., "q_proj", "gate_proj") that peft will apply LoRA adapters to; must be enumerated per model architecture.

**PeftModel**: The peft wrapper class for a base model with loaded adapters; loaded via `PeftModel.from_pretrained(base_model, adapter_path)`.

**adapter_model.safetensors**: The file saved by `model.save_pretrained()` on a PeftModel; contains only A and B matrices for all LoRA layers, typically 10–100MB.

**adapter_config.json**: Metadata file saved alongside the adapter weights; contains the LoraConfig (rank, alpha, target_modules, etc.) needed to reconstruct the adapter structure.

**merge_and_unload**: A PeftModel method that computes W_merged = W + BA * (alpha/r) in-place, removes the LoRA structure, and returns a standard HuggingFace model — used before full deployment.

**Grouped Query Attention (GQA)**: An attention variant where multiple query heads share one key/value head; affects k_proj and v_proj dimensions in some models (e.g., Llama 3, Qwen2.5-7B).

**lora_dropout**: Dropout probability applied to the input of the LoRA path before multiplying by lora_A; adds regularization. Typical values: 0.05–0.1.

**bias strategy**: Controls whether bias vectors are adapted in LoRA; options: "none" (frozen), "all" (all biases trained), "lora_only" (only targeted layers' biases trained).
