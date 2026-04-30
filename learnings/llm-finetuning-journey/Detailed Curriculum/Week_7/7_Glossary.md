# Week 7 — Glossary

**AutoClass**: HuggingFace pattern for loading models/tokenizers by string ID; reads `config.json` and instantiates the correct architecture automatically.

**AutoModelForCausalLM**: AutoClass for decoder-only language models (GPT-2, LLaMA, Qwen); used for text generation tasks.

**AutoModelForSeq2SeqLM**: AutoClass for encoder-decoder models (T5, BART); used for sequence-to-sequence tasks like translation.

**from_pretrained**: HuggingFace method to load weights, config, and tokenizer from a Hub repository or local path; caches results automatically.

**HuggingFace Hub**: Repository hosting ~500K+ public models, datasets, and Spaces; accessed via API token.

**transformers**: HuggingFace library wrapping PyTorch with model implementations, training utilities (`Trainer`), and inference pipelines.

**datasets**: HuggingFace library for loading, processing, and caching datasets in Arrow/Parquet format.

**Arrow format**: Columnar in-memory and on-disk data format used by the `datasets` library; supports memory-mapping for large datasets.

**push_to_hub**: Method to upload a model, tokenizer, or dataset to your HuggingFace account.

**save_to_disk / load_from_disk**: Local save/load for `datasets.Dataset` objects in Arrow format.

**batched=True**: Parameter for `Dataset.map()` that passes a dict of lists to the function, enabling vectorized processing (e.g., fast tokenization).

**labels (causal LM)**: Tensor of target token IDs for next-token prediction; typically `input_ids` shifted by one, with padding positions set to -100.

**-100**: PyTorch `nn.CrossEntropyLoss` ignore_index; positions labeled -100 contribute zero loss and zero gradient.

**greedy decoding**: Text generation that always selects the highest-probability next token; deterministic but locally suboptimal.

**top-k sampling**: Restricts sampling pool to the k highest-probability tokens at each step; k is fixed regardless of distribution shape.

**top-p (nucleus) sampling**: Restricts sampling to the smallest set of tokens whose cumulative probability exceeds p; pool size adapts to model confidence.

**temperature**: Scalar dividing logits before softmax; < 1 sharpens the distribution (more deterministic), > 1 flattens it (more random).

**attention_mask**: Binary tensor where 1 = real token, 0 = padding; tells the model which positions to attend to and which to ignore.

**pad_token_id**: Token ID used for padding shorter sequences in a batch to equal length; must be set for models that don't define it by default (e.g., GPT-2).
