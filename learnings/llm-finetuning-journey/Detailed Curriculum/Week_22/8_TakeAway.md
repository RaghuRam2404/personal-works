# Week 22 TakeAway — Evaluating a Language Model

**One-liner:** Perplexity = exp(mean CE loss); diagnose with samples; your 50M model is not your fine-tuning base — Qwen2.5-Coder-7B is.

---

## Key Formulas

```python
import math

# Perplexity from per-batch losses
avg_ce_loss = sum(losses) / len(losses)   # average over batches
perplexity = math.exp(avg_ce_loss)        # NOT mean(exp(losses))

# Compute perplexity manually
loss = F.cross_entropy(logits.view(-1, V), targets.view(-1))
ppl = math.exp(loss.item())
```

---

## Key Code Pattern — Text Generation

```python
def generate(model, ids, max_new=150, temp=0.8, top_k=50):
    model.eval()
    x = torch.tensor(ids).unsqueeze(0)
    for _ in range(max_new):
        x_cond = x[:, -1024:]
        with torch.no_grad():
            logits, _ = model(x_cond)
        logits = logits[:, -1, :] / temp
        if top_k:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float('-inf')
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)
        x = torch.cat([x, nxt], dim=1)
    return x[0, len(ids):]
```

---

## Perplexity Reference Table

| Model | Params | Tokens | PPL (web text) |
|---|---|---|---|
| Your 50M | 56M | 2B | 25–40 |
| GPT-2 small | 117M | 40B | ~18 |
| GPT-2 medium | 345M | 40B | ~14 |
| GPT-Neo 1.3B | 1.3B | ~300B | ~8 |

---

## Decision Rules

- PPL > 50 on FineWeb val → investigate data, tokenizer, or training completeness
- PPL < 15 on FineWeb val → check for train/val data overlap (contamination)
- Temperature 0.7–0.9 → best for coherent generation on a small model
- Temperature < 0.3 → repetition loops likely; use repetition penalty
- top_k = 50 → good default; lower for more conservative SQL generation
- To compare models with different tokenizers → use bits-per-character (BPC) not PPL

---

## Numbers to Remember

| Quantity | Value |
|---|---|
| exp(3.5) | 33.1 |
| exp(3.2) | 24.5 |
| exp(4.0) | 54.6 |
| exp(2.8) | 16.4 |
| Random model PPL (vocab=32K) | 32,000 |

---

## Red Flags

- mean(exp(per_batch_loss)) instead of exp(mean(per_batch_loss)) → biased high
- Perplexity computed with model.train() mode → dropout active → artificially inflated
- All 12 samples have identical structure → temperature too low
- val PPL much lower than train PPL → val set is inside training data (data leak)
