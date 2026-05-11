# Week 58 Glossary

**SFTTrainer:** The TRL class implementing supervised fine-tuning with proper handling of chat templates, sequence packing, and loss masking.

**SFTConfig:** The configuration class for SFTTrainer; extends TrainingArguments with SFT-specific options like `max_seq_length` and `dataset_text_field`.

**Completion-only loss:** A training mode where cross-entropy loss is computed only on the assistant/completion tokens, not on the prompt/instruction tokens; implemented via DataCollatorForCompletionOnlyLM.

**DataCollatorForCompletionOnlyLM:** A TRL collator that masks all tokens before the response template to -100 (ignored in loss) so the model only learns to predict assistant responses.

**Response template:** The string token sequence marking the start of an assistant response in a chat-formatted training example (e.g., `<|im_start|>assistant\n` for Qwen2.5).

**Training loss:** The cross-entropy loss computed on the training batch; smoothly decreasing training loss is a necessary but not sufficient condition for good model performance.

**Validation loss:** Cross-entropy loss computed on held-out validation examples; increasing validation loss indicates overfitting.

**Execution accuracy:** The fraction of model-generated SQL queries that execute without error and return the correct result set; the primary metric for text-to-SQL evaluation.

**Early stopping:** Halting training when a monitored metric (validation loss, execution accuracy) fails to improve for a specified number of evaluation intervals.

**LoRA rank (r):** The rank of the low-rank decomposition matrices in LoRA; higher rank captures more capacity but uses more parameters and memory. Typical SFT values: 16–64.

**lora_alpha:** The LoRA scaling factor; `effective_lr = lr * lora_alpha / r`. Setting `lora_alpha = 2*r` gives an effective scaling of 2.0.

**load_best_model_at_end:** A HuggingFace TrainingArguments flag that loads the checkpoint with the best `metric_for_best_model` value at the end of training, rather than the final checkpoint.
