# Week 14 Glossary

**LLaMA (Large Language Model Meta AI)**: Meta AI's open-weights decoder-only LLM family; first high-quality open-source LLM (Feb 2023).

**Base model**: A pretrained language model trained only on next-token prediction; no instruction tuning or RLHF applied.

**Instruction-tuned model**: A base model further fine-tuned on instruction-following data (e.g., LLaMA 2-Chat, Alpaca).

**num_key_value_heads**: HuggingFace config parameter controlling the number of GQA KV heads; if < num_attention_heads, GQA is active.

**num_key_value_groups**: `num_attention_heads / num_key_value_heads`; the number of query heads per KV head in GQA.

**repeat_kv**: HF function that broadcasts KV heads from n_kv_heads to n_heads before attention; no learned parameters.

**rope_theta (base)**: The base frequency for RoPE; higher values (500000) enable longer-context position distinction.

**Over-training**: Training beyond the compute-optimal number of tokens; produces a smaller model with equivalent quality — better for inference.

**Tokenizer vocabulary size**: Number of tokens the tokenizer can represent; LLaMA 3 uses 128k (BPE), LLaMA 1/2 use 32k.

**BPE (Byte-Pair Encoding)**: Tokenization algorithm that merges frequent character pairs; LLaMA 3's 128k vocabulary uses a BPE tokenizer.

**Compute-optimal training**: Training with the optimal balance of model size and training tokens for a given compute budget (Chinchilla law).

**Gradient checkpointing**: Trading compute for memory — recomputes activations during backward pass instead of storing them; enables training large models on limited VRAM.

**Weight tying (absent in LLaMA)**: LLaMA keeps embedding and LM head weights separate, unlike GPT-2; simplifies the architecture at the cost of a small parameter overhead.
