# Week 23 Assignment Solutions

## Task 4 — Key Snippet: Manual Log-Likelihood Evaluation

```python
import torch
import torch.nn.functional as F
from datasets import load_dataset

def score_option(model, tokenizer, context, option, device='cuda'):
    """Compute log P(option | context) using the model."""
    full_text = context + " " + option
    ctx_ids = tokenizer.encode(context).ids
    full_ids = tokenizer.encode(full_text).ids
    option_ids = full_ids[len(ctx_ids):]

    x = torch.tensor(full_ids[:-1], dtype=torch.long).unsqueeze(0).to(device)
    y = torch.tensor(full_ids[1:],  dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, _ = model(x)

    # Extract logits corresponding to option positions
    option_start = len(ctx_ids) - 1  # -1 because x is shifted by 1
    option_logits = logits[0, option_start:option_start+len(option_ids), :]
    option_targets = y[0, option_start:option_start+len(option_ids)]

    log_probs = F.log_softmax(option_logits, dim=-1)
    option_log_prob = log_probs.gather(1, option_targets.unsqueeze(1)).sum()
    # Normalize by length to avoid bias toward short options
    return option_log_prob.item() / len(option_ids)

ds = load_dataset("Rowan/hellaswag", split="validation")
correct = 0
for i, ex in enumerate(ds):
    if i >= 5: break
    ctx = ex["ctx"]
    endings = ex["endings"]
    label = int(ex["label"])
    scores = [score_option(model, tokenizer, ctx, e) for e in endings]
    pred = scores.index(max(scores))
    correct += (pred == label)
    print(f"Q{i+1}: pred={pred} label={label} scores={[f'{s:.3f}' for s in scores]}")

print(f"Accuracy on 5 examples: {correct}/5 = {correct/5:.0%}")
```

**Expected output for 50M model:**
- 1–3 out of 5 correct (20–60% accuracy on a tiny sample; high variance)
- GPT-2 would get 1–2 out of 5 on average (31.6% over 10K examples)
- Your model may be similar or slightly worse

**Common gotchas:**
- Off-by-one in `option_start` — the model's logit at position i predicts token at position i+1
- Not normalizing by option length → biased toward shorter options (common in HellaSwag)
- Using `tokenizer.encode(full_text).ids` where the tokenizer joins context + option differently than concatenation — always verify with a manual check

---

## Task 2 — Common Issues Running lm-eval on Your Custom Model

**Issue 1: Custom tokenizer not recognized**

lm-evaluation-harness expects a HuggingFace-compatible tokenizer. If you trained a custom `ByteLevelBPETokenizer`, convert it:

```python
from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2TokenizerFast

bpe_tok = ByteLevelBPETokenizer.from_file("tokenizer/vocab.json", "tokenizer/merges.txt")
# Wrap in transformers-compatible class
hf_tok = GPT2TokenizerFast(tokenizer_object=bpe_tok)
hf_tok.add_special_tokens({'eos_token': '<|endoftext|>'})
hf_tok.save_pretrained("tokenizer_hf/")
```

**Issue 2: Model architecture not GPT2LMHeadModel**

lm-eval uses `AutoModelForCausalLM`. If your model is custom, either:
- Convert it to a GPT2LMHeadModel as in Week 21 solution
- Implement a custom `lm_eval.models.HuggingFaceModel` subclass

The simplest path: convert your checkpoint to HuggingFace format in Week 21 and evaluate from the Hub.

---

## Expected Comparison Table

| Benchmark | Random | GPT-2 (117M, 40B tok) | Your 50M (2B tok) |
|---|---|---|---|
| HellaSwag 0-shot | 25% | 31.6% | 26–30% |
| ARC-Easy 0-shot | 25% | 43.2% | 28–35% |
| ARC-Challenge 0-shot | 25% | 25.9% | 24–28% |
| Val PPL (web text) | ~32K | ~18 | 25–40 |

Your model should be closer to GPT-2 than to random on HellaSwag and ARC-Easy. If you score below 25% (worse than random), investigate: your log-likelihood computation may have a sign error.
