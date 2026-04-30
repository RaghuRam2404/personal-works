# Week 6 — Glossary

**Tokenization**: The process of splitting raw text into discrete units (tokens) that serve as input to a neural network.

**Subword tokenization**: Splitting words into frequent subword units; balances vocabulary size, sequence length, and coverage of rare/unknown words.

**BPE (Byte-Pair Encoding)**: Subword tokenization algorithm that iteratively merges the most frequent adjacent byte/token pair into a new vocabulary token.

**Merge rule**: An ordered pair of tokens and their merged result (e.g., `('S','E') → 256`); the BPE vocabulary is a sorted list of merge rules.

**Byte-level BPE**: BPE applied to individual bytes (0–255) rather than characters; guarantees no OOV tokens for any Unicode input.

**Pre-tokenization**: Splitting text into chunks (e.g., by word boundary) before applying BPE, using a regex pattern; prevents merges across word boundaries.

**Vocabulary size**: Total number of tokens (byte base + merges) in a tokenizer; GPT-4 = 100,277; Qwen2.5-Coder ≈ 150,000.

**tiktoken**: OpenAI's fast BPE tokenizer library (Rust backend); implements cl100k_base and other encodings.

**WordPiece**: BERT's tokenization algorithm; uses likelihood-based merges and marks non-initial pieces with `##`.

**SentencePiece**: Tokenization library supporting BPE and Unigram; treats spaces as regular characters (prefix `▁`); used by LLaMA, T5, Qwen.

**OOV (Out-of-Vocabulary)**: A token not present in the vocabulary; byte-level BPE eliminates OOV by design.

**Special token**: A token added to the vocabulary for structural purposes: `<|endoftext|>`, `<|im_start|>`, `<PAD>`, `<|sql_end|>`.

**Attention mask**: Binary tensor indicating which positions are real tokens (1) vs. padding (0); must be passed to the model to prevent padding from affecting attention.

**Token ID**: The integer index of a token in the vocabulary; what the embedding table actually indexes.

**round-trip**: Property that `decode(encode(text)) == text`; a basic correctness check for any tokenizer.

**add_special_tokens**: HuggingFace tokenizer parameter controlling whether BOS/EOS special tokens are added to the encoding.
