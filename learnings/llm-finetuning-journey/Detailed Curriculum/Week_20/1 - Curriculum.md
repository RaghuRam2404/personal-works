# Week 20 — Pretraining Setup: Codebase, Model Size, and Tokenizer

## Learning Objectives

By the end of this week, you will be able to:

- Set up a clean pretraining codebase based on nanoGPT
- Justify a ~50M parameter GPT configuration mathematically
- Train a Byte-Pair Encoding (BPE) tokenizer from scratch using HuggingFace tokenizers
- Build a data pipeline that streams FineWeb-Edu and converts it to tokenized `.bin` files
- Verify that your codebase is wired up correctly with a short sanity-check training run

---

## Concepts

### Structuring a Pretraining Codebase

You will build on [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT), but you should not simply copy-paste it. The goal is to understand every line. A minimal pretraining codebase needs:

1. **Model definition** (`model.py`): GPT transformer, tokenizer embedding, positional embedding, stack of transformer blocks, language modeling head
2. **Data pipeline** (`data.py`): Streaming from HuggingFace datasets, tokenization, chunking into context-length windows, writing to memory-mapped `.bin` files
3. **Training loop** (`train.py`): DataLoader, forward pass, loss, backward, AdamW optimizer, LR scheduler (cosine with warmup), W&B logging, checkpoint saving
4. **Config** (`config.py`): All hyperparameters in one place; no magic numbers in `train.py`
5. **Evaluation** (`eval.py`): Validation loop, perplexity computation, text generation

The key structural principle: keep the model definition separate from the training logic. This is how every production training codebase is organized.

### Choosing Model Size: ~50M Parameters

A GPT transformer has the following parameter count (excluding embeddings):

```
Attention:  4 × d_model²        (Q, K, V, output projections)
FFN:        8 × d_model²        (typically 4 × d_model expansion, 2 matrices)
LayerNorm:  4 × d_model         (negligible)
Per layer:  12 × d_model²       (approximate)
Total:      n_layers × 12 × d_model²
```

Plus embeddings:
```
Token embedding:    vocab_size × d_model
Position embedding: context_len × d_model
```

**Targeting 50M parameters:**

You want `n_layers × 12 × d_model² ≈ 50M`. Reasonable configurations:

| n_layers | d_model | n_heads | Params (approx) |
|---|---|---|---|
| 6 | 512 | 8 | 18.9M |
| 8 | 512 | 8 | 25.2M |
| 6 | 768 | 12 | 42.5M |
| 8 | 768 | 12 | 56.6M |
| 6 | 1024 | 16 | 75.5M |

**Recommended config for this project:** 8 layers, d_model=768, 12 heads → ~56M params. This is similar to GPT-2 small (117M) but smaller; it fits easily on any A100 and trains fast.

**Verifying your parameter count:**

```python
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

n_params = count_params(model)
print(f"Parameters: {n_params/1e6:.1f}M")
```

### Training a BPE Tokenizer

You will train a BPE tokenizer on your FineWeb-Edu sample rather than using GPT-2's tokenizer. This teaches you the mechanics and gives you ownership of the vocabulary.

**Why train your own tokenizer?**
- Practice the mechanics for your Phase 6 domain tokenizer
- Understand what vocabulary size means in memory terms
- See how tokenization affects model quality (bad tokenizer → higher effective perplexity)

**HuggingFace tokenizers API:**

```python
from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()
tokenizer.train_from_iterator(
    iterator=text_iterator(),    # generator that yields strings
    vocab_size=32000,
    min_frequency=2,
    special_tokens=["<|endoftext|>", "<|pad|>"]
)
tokenizer.save_model("tokenizer/")
```

**Vocabulary size considerations:**

| Vocab size | Embedding memory (d_model=768) | Coverage | Tokens per English word |
|---|---|---|---|
| 8,000 | 24MB | Limited | ~1.5 |
| 32,000 | 98MB | Good | ~1.1 |
| 50,257 (GPT-2) | 154MB | Excellent | ~1.0 |
| 100,000 | 307MB | Excellent | <1.0 |

For a 50M-param model, use vocab_size=32,000. The embedding layer adds only 24M parameters but doesn't affect training dynamics much.

**Training data for tokenizer:** Use a 200M-word (roughly 150MB text) sample of FineWeb-Edu. You do not need the full dataset for the tokenizer.

### Building the Data Pipeline

The efficient approach: tokenize all data upfront into `.bin` files (memory-mapped uint16 arrays), then sample from them during training.

```python
import numpy as np
from datasets import load_dataset

def prepare_data(output_path, tokenizer, max_tokens=100_000_000):
    ds = load_dataset("HuggingFaceFW/fineweb-edu",
                      name="sample-10BT", split="train", streaming=True)
    all_tokens = []
    for doc in ds:
        tokens = tokenizer.encode(doc["text"]).ids
        tokens.append(tokenizer.token_to_id("<|endoftext|>"))
        all_tokens.extend(tokens)
        if len(all_tokens) >= max_tokens:
            break

    arr = np.array(all_tokens, dtype=np.uint16)
    arr.tofile(output_path)
    print(f"Saved {len(arr)} tokens to {output_path}")
```

**Memory-mapped loading during training:**

```python
class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, path, block_size):
        self.data = np.memmap(path, dtype=np.uint16, mode='r')
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = torch.from_numpy(
            self.data[idx:idx+self.block_size+1].astype(np.int64)
        )
        x, y = chunk[:-1], chunk[1:]
        return x, y
```

### The Model Architecture

Your GPT implementation should have:

```python
class CausalSelfAttention(nn.Module):
    # Multi-head attention with causal mask
    # Use flash attention if available: F.scaled_dot_product_attention(q, k, v, is_causal=True)

class MLP(nn.Module):
    # Linear(d_model, 4*d_model) → GELU → Linear(4*d_model, d_model)

class TransformerBlock(nn.Module):
    # LayerNorm → Attention → residual
    # LayerNorm → MLP → residual

class GPT(nn.Module):
    # Token embedding + position embedding
    # n_layers × TransformerBlock
    # Final LayerNorm
    # Linear head (d_model → vocab_size), weight-tied to embedding
```

**Weight tying:** Tie the input embedding weights to the output LM head projection. This reduces parameter count and improves training stability.

### Sanity Checks Before Week 21

Before starting the full training run, verify:
1. `count_params(model)` shows ~50–60M parameters
2. A single batch forward pass completes without error
3. Initial loss is approximately `log(vocab_size)` = log(32000) ≈ 10.4 (random initialization)
4. Loss decreases over 100 steps on a small toy dataset
5. W&B logging is connected and shows a training curve

---

## Connections

**Prior weeks:** Week 18 (FineWeb data) feeds your data pipeline. Week 19 (Accelerate) will wrap your training loop. Week 17 (Chinchilla) justified your 50M parameter count.

**Weeks 21–22:** This week's setup is the prerequisite. Do not start Week 21 without completing all sanity checks above.

---

## Common Misconceptions

- **"I should use GPT-2's tokenizer to save time."** You can — and for speed it is fine — but you miss the learning opportunity. Train your own for this project.
- **"I need to implement rotary embeddings (RoPE) for this."** Not for a 50M model. Absolute positional embeddings are fine at context_len=1024. RoPE becomes important at context_len>4096.
- **"Flash Attention is optional."** For small models on Colab A100, it is optional. But `F.scaled_dot_product_attention` with `is_causal=True` is one line and gives 20–30% speedup — use it.
- **"I should maximize context length for better language modeling."** Context length drives memory quadratically. At 50M params, use context_len=1024 or 512. Focus on training many tokens, not on long context.

---

## Time Allocation (6–8 hrs)

- 1h: Fork nanoGPT, study the model definition, write your own version
- 1h: Train BPE tokenizer on FineWeb-Edu sample
- 1.5h: Write data pipeline (prepare_data script + TokenDataset)
- 2h: Write training loop with Accelerate + W&B logging
- 1h: Run sanity checks, verify loss starts at ~log(vocab_size)
- 0.5h: Commit everything with `week-20-pretrain-setup`
