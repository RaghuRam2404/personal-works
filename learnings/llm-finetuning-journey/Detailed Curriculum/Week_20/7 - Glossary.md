# Week 20 Glossary — Pretraining Setup

**BPE (Byte-Pair Encoding)**: A subword tokenization algorithm that iteratively merges the most frequent character pairs; used by GPT-2, GPT-4, Llama, Qwen, and most modern LLMs.

**ByteLevelBPETokenizer**: HuggingFace tokenizers implementation of BPE that operates on raw UTF-8 bytes, enabling coverage of any Unicode character without unknown tokens.

**Vocabulary size**: The number of unique tokens the tokenizer can produce; determines the size of the embedding matrix and LM head.

**Memory-mapped file (memmap)**: A file-backed array where the OS loads only the portions needed into RAM; used to read tokenized training data without loading all tokens at once.

**context_len (context window)**: The maximum number of tokens the model can attend to in a single forward pass; limited by positional embedding size.

**Pre-LayerNorm (pre-LN)**: Applying Layer Normalization before each sub-layer (attention, FFN) rather than after; more stable gradients than post-LN.

**Weight tying**: Sharing the weights of the token embedding matrix and the output LM head projection; reduces parameter count and often improves perplexity.

**LM head**: The final linear layer that projects from d_model to vocab_size to produce logits for next-token prediction.

**Flash Attention**: An IO-aware attention algorithm that fuses the attention computation to minimize GPU memory bandwidth usage; available via `F.scaled_dot_product_attention` in PyTorch 2.0+.

**Cosine LR schedule**: A learning rate schedule that decreases from max_lr to min_lr following a cosine curve; standard for language model pretraining.

**Linear warmup**: An initial phase of LR schedule where the learning rate increases linearly from 0 to max_lr over a fixed number of steps.

**Gradient clipping**: Scaling gradients to have a maximum norm (typically 1.0); prevents exploding gradients during early training.

**AdamW**: Adam optimizer with decoupled weight decay; the standard optimizer for transformer pretraining; use beta1=0.9, beta2=0.95 for LM pretraining (not 0.999).

**block_size**: The chunk size (in tokens) fed to the model during training; typically set equal to context_len.

**uint16**: A 16-bit unsigned integer (0–65535); used to store token IDs efficiently, halving disk space vs float32.
