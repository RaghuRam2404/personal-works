# Week 22 Assignment Solutions

## Task 1 — Key Snippet: Perplexity Computation

```python
import math, torch
import numpy as np

@torch.no_grad()
def compute_perplexity(model, val_path, block_size=1024, n_batches=200, device='cuda'):
    model.eval()
    data = np.memmap(val_path, dtype=np.uint16, mode='r')

    losses = []
    for i in range(n_batches):
        start = i * block_size
        if start + block_size + 1 > len(data):
            break
        chunk = torch.from_numpy(data[start:start+block_size+1].astype(np.int64))
        x = chunk[:-1].unsqueeze(0).to(device)
        y = chunk[1:].unsqueeze(0).to(device)
        _, loss = model(x, y)
        losses.append(loss.item())

    avg_loss = sum(losses) / len(losses)
    std_loss = (sum((l - avg_loss)**2 for l in losses) / len(losses))**0.5
    ppl = math.exp(avg_loss)
    ppl_upper = math.exp(avg_loss + 1.96 * std_loss / len(losses)**0.5)
    ppl_lower = math.exp(avg_loss - 1.96 * std_loss / len(losses)**0.5)

    print(f"Val CE loss: {avg_loss:.4f} ± {std_loss:.4f}")
    print(f"Val perplexity: {ppl:.1f} (95% CI: {ppl_lower:.1f}–{ppl_upper:.1f})")
    return ppl
```

**Common gotchas:**
- Forgetting `.astype(np.int64)` — uint16 out of range for `nn.Embedding` on GPU
- Computing `exp(mean(losses))` vs. `mean(exp(losses))` — the former is correct; these differ significantly
- Not setting `model.eval()` — dropout or other stochastic layers active during eval will inflate perplexity

---

## Task 2 — Key Snippet: Generation with Top-k and Temperature

```python
def generate(model, tokenizer, prompt, max_new=150, temp=0.8, top_k=50):
    model.eval()
    ids = tokenizer.encode(prompt).ids
    x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(max_new):
        x_cond = x[:, -1024:]  # context window cap
        with torch.no_grad():
            logits, _ = model(x_cond)
        logits = logits[:, -1, :] / temp  # (1, vocab_size)

        # Top-k filter
        if top_k is not None:
            topk_vals, _ = torch.topk(logits, top_k)
            logits[logits < topk_vals[:, -1:]] = float('-inf')

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        x = torch.cat([x, next_id], dim=1)

    new_ids = x[0, len(ids):].tolist()
    return tokenizer.decode(new_ids)
```

**Expected sample quality (50M model, FineWeb-Edu training):**
- Prompts 1–4: Somewhat coherent for 20–40 tokens; degradation after that
- Prompts 5–8: Likely produces partially valid SQL syntax, but wrong table names/columns
- Prompt 11–12: May or may not complete "The capital of France is Paris" — 50M models often fail this

**Common gotchas:**
- Using `tokenizer.decode` with byte-level BPE — ensure the library handles the byte fallback correctly
- Infinite repetition at low temperature → add a repetition penalty or use higher temperature
- Context window cap (1024) — if you do not cap, you will get index out of range errors

---

## Task 4 — Section 6 Template (Fine-Tuning Readiness)

**Model answer for section 6:**

Your 50M model achieves perplexity ~30 on general English web text. For fine-tuning on PostgreSQL/TimescaleDB SQL, this model has two fundamental limitations:

1. It has almost no SQL-specific knowledge. The generated SQL samples (prompts 5–8) show partial syntax awareness but wrong semantics, missing JOINs, and invented column names. Fine-tuning a model that already "knows" code and SQL on domain-specific examples amplifies that knowledge — fine-tuning a model that barely knows SQL teaches from near scratch.

2. 50M parameters cannot hold enough SQL knowledge to beat GPT-4. Even with perfect fine-tuning, a 50M model has ~1000× fewer parameters than GPT-4. The information capacity limits what it can learn.

The correct choice for Phase 6 is Qwen2.5-Coder-7B: pre-trained on 5.5T tokens of code-heavy data, already produces valid SQL, and has the capacity to learn complex PostgreSQL idioms. Your fine-tuning effort will be applied to a model that starts from a position of strength, not from scratch.

The 50M pretraining experience was not wasted — it taught you the entire pipeline, debugging skills, and evaluation skills that you will apply when fine-tuning the 7B model.
