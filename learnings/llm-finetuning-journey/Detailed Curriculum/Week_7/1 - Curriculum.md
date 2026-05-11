# Week 7 — HuggingFace Ecosystem Onboarding

## Learning Objectives

By the end of this week, you will be able to:

- Load any model from the HuggingFace Hub using `AutoModel` and `AutoTokenizer`, run inference, and inspect the output logits.
- Use the `datasets` library to load, filter, map, and split standard datasets — including Spider.
- Understand and use the `attention_mask` and `labels` fields required for language model training with HuggingFace `Trainer`.
- Push a tokenized dataset to your HuggingFace account as a private dataset artifact.
- Identify which components of the HuggingFace stack you will use in each future phase and how they relate to raw PyTorch.
- Explain the `AutoClass` pattern, `PreTrainedModel`, and `PreTrainedTokenizerFast` class hierarchy.

---

## Concepts

### The HuggingFace Stack

HuggingFace consists of four main libraries you will use throughout this course:

| Library | Purpose | First used |
|---|---|---|
| `transformers` | Models, tokenizers, `Trainer`, pipelines | Week 7 |
| `datasets` | Loading, processing, caching datasets | Week 7 |
| `peft` | LoRA, adapters, efficient fine-tuning | Phase 4 |
| `trl` | SFT, DPO, PPO trainers for alignment | Phase 5 |

Each library is a high-level wrapper over PyTorch. You are not replacing your PyTorch knowledge — you are building a layer on top of it. When something breaks, you will need to go back to raw PyTorch to debug.

### AutoClass Pattern

`AutoTokenizer.from_pretrained("model_id")` and `AutoModelForCausalLM.from_pretrained("model_id")` are the entry points to the Hub. The `Auto` prefix means HuggingFace reads the model's `config.json`, determines the architecture (e.g., Qwen2, Llama, GPT2), and instantiates the correct class automatically.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tok   = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
```

Key model classes for your domain:
- `AutoModelForCausalLM` — decoder-only LMs (GPT-2, Qwen, LLaMA). Used for text generation.
- `AutoModelForSeq2SeqLM` — encoder-decoder (T5, BART). Used for sequence-to-sequence tasks.
- `AutoModel` — base model, no task-specific head. Used when you add your own head.

### Model Hub Vocabulary

Every model on the Hub has:
- A **model ID** in the form `"org/model-name"` (e.g., `"Qwen/Qwen2.5-Coder-7B-Instruct"`).
- A **config.json** describing the architecture hyperparameters.
- **model.safetensors** (or `.bin`) files with the weights.
- A **tokenizer** (multiple files: `tokenizer.json`, `vocab.json`, `merges.txt` or `sentencepiece.model`).

The `from_pretrained` call downloads and caches all of these in `~/.cache/huggingface/hub/` by default.

### The `datasets` Library

```python
from datasets import load_dataset

# Load Spider from HuggingFace Hub
ds = load_dataset("spider")   # dict with 'train', 'validation' splits
ds['train'][0]                # first example: {'db_id', 'query', 'question', ...}

# Map / filter
def add_prompt(example):
    example['text'] = f"Question: {example['question']}\nSQL: {example['query']}"
    return example

ds = ds.map(add_prompt)
ds = ds.filter(lambda x: len(x['text']) < 512)
ds = ds.remove_columns(['db_id', 'question', 'query'])
```

The `datasets` library uses Arrow for columnar storage and memory-mapping — datasets that don't fit in RAM can still be processed with `.map()`. Processed datasets are cached on disk automatically (keyed by the function hash), so `.map()` is only slow the first time.

**Tokenizing a dataset:**

```python
def tokenize(examples):
    return tok(examples['text'], truncation=True, max_length=256)

tokenized = ds.map(tokenize, batched=True, remove_columns=['text'])
# Result has 'input_ids', 'attention_mask' columns
```

Use `batched=True` for tokenization — it is dramatically faster than processing one example at a time (calls the tokenizer on a list at once).

### `attention_mask` and `labels` for Causal LM Training

When fine-tuning a causal LM:
- `input_ids`: the token sequence, shape `(batch, seq_len)`.
- `attention_mask`: 1 for real tokens, 0 for padding, shape `(batch, seq_len)`.
- `labels`: for causal LM, `labels = input_ids` shifted by one position; the model predicts `input_ids[t]` from `input_ids[:t]`. In practice: `labels = input_ids.clone()`, then set padding positions to -100 (HuggingFace's ignore index).

```python
# Labels for causal LM
labels = batch['input_ids'].clone()
labels[batch['attention_mask'] == 0] = -100  # ignore padding in loss
```

The `-100` convention is HuggingFace's: `nn.CrossEntropyLoss` ignores positions where `label == -100`. This is critical — without it, the model tries to predict the padding tokens, which produces incorrect loss and gradients.

For text-to-SQL fine-tuning (Phase 4), you will also mask the prompt portion of the input (set prompt tokens to -100) so the model only learns to predict the SQL, not the question or schema. This week, just understand the pattern.

### Running Inference

```python
model.eval()
with torch.no_grad():
    inputs = tok("SELECT ", return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tok.eos_token_id,
    )
    print(tok.decode(output[0], skip_special_tokens=True))
```

Inspect logits directly:
```python
with torch.no_grad():
    out = model(**inputs)
    logits = out.logits          # (batch, seq_len, vocab_size)
    probs  = logits.softmax(-1)
    top5   = probs[0, -1].topk(5)  # top 5 next token predictions
    print(tok.convert_ids_to_tokens(top5.indices.tolist()))
```

### Pushing to HuggingFace Hub

```python
from huggingface_hub import login
login()  # or set HF_TOKEN env var

tokenized.push_to_hub("your_username/spider-tokenized-qwen25", private=True)
```

This creates a private dataset repo on your HuggingFace account that you can access from any machine — including Colab and RunPod. This is the workflow for sharing training data between your Mac and remote compute environments.

---

## Connections

**Builds on:** Week 6's tokenization understanding. Week 5's training loop.

**Unlocks:** Everything in Phases 2–6 uses the HuggingFace stack. Week 8's capstone uses `AutoModelForCausalLM` and `datasets`. Phase 4 introduces `peft` and `Trainer`. Phase 5 introduces `trl`.

---

## Common Misconceptions and Pitfalls

- **"`from_pretrained` downloads the model on every run."** No — HuggingFace caches in `~/.cache/huggingface/hub`. The model is only downloaded once (or when you force refresh with `force_download=True`).
- **"Using `Trainer` means I don't need to understand the training loop."** Wrong. `Trainer` is a wrapper around the same loop you wrote in Week 1. When it breaks (and it will), you need to know which part of the loop the error corresponds to.
- **"Labels should always equal input_ids for causal LM training."** Only for pre-training. For instruction fine-tuning (Phase 4), you mask the prompt tokens to -100. For preference data (Phase 5), the label format is completely different.
- **"Padding is harmless."** Unmasked padding tokens in `labels` will cause the model to try to predict `<PAD>` at those positions, corrupting the loss. Always set padding tokens to -100 in labels.

---

## Time Allocation (6–8 hours this week)

| Activity | Time |
|---|---|
| Read HuggingFace LLM Course Chapters 1–3, do all exercises | 2 h |
| Load distilgpt2, run inference, inspect logits | 45 min |
| Load Spider from datasets hub, explore structure | 30 min |
| Tokenize Spider with Qwen2.5-Coder tokenizer, push to Hub | 1.5 h |
| Push tokenized dataset to your HuggingFace account | 30 min |
| Explore `model.generate()` parameters (temperature, top-k, top-p) | 45 min |
| Journal + commit | 30 min |
