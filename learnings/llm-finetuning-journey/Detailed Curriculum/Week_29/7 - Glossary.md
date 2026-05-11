# Week 29 Glossary

**SFTTrainer**: HuggingFace TRL class that wraps `Trainer` for supervised fine-tuning, handling input masking and packing automatically.

**SFTConfig**: Configuration dataclass for `SFTTrainer`; extends `TrainingArguments` with SFT-specific settings like `max_seq_length` and `packing`.

**Chat template**: Model-specific format that converts a list of `{"role": ..., "content": ...}` messages into a flat token sequence with special role-delimiter tokens.

**ChatML format**: Qwen2.5's chat template using `<|im_start|>role` and `<|im_end|>` as message delimiters; used by many modern open models.

**Packing**: Concatenating multiple short training examples into one sequence up to `max_seq_length` to eliminate padding waste and increase GPU utilization.

**Label masking**: Setting loss labels to -100 for prompt tokens so cross-entropy loss is computed only over response tokens; implemented automatically by `SFTTrainer`.

**Gradient clipping**: Rescaling gradients to have maximum norm `max_grad_norm` (typically 1.0) to prevent gradient explosion during training.

**`add_generation_prompt`**: A tokenizer argument that appends the start-of-assistant marker to the prompt at inference time, signaling the model to begin generating a response.

**Eval loss**: Cross-entropy loss on held-out validation examples; rising eval loss when train loss falls is the signal for overfitting.

**Model card**: A README file on HuggingFace Hub that describes the model, its training data, intended use, and how to run inference.

**`device_map="auto"`**: A HuggingFace argument that automatically distributes model layers across available GPUs (and CPU if needed) for loading large models.

**Gradient accumulation**: Performing multiple forward/backward passes before updating weights, effectively simulating a larger batch size than fits in memory.
