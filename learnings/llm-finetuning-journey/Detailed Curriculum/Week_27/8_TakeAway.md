# Week 27 TakeAway — Phase 3 Gate

**One-liner:** Phase 3 Gate: verify evidence, not intuition, before entering fine-tuning.

---

## Gate Criteria at a Glance

| Criterion | Minimum Bar | Evidence Type |
|---|---|---|
| 50M LM trained | val loss < 4.0 | W&B URL + checkpoint path |
| Perplexity computable | 15 ≤ PPL ≤ 60 | `eval.py` output |
| v1 dataset complete | ≥5K examples, ≥98% SQL valid | HuggingFace Hub URL |
| Can read LLM reports | Explain GQA, MoE, FIM cold | Filled `week-24-sota-comparison.md` |

---

## Key Formulas to Recall Without Notes

```
# Perplexity from cross-entropy loss
PPL = exp(mean_CE_loss)

# Chinchilla optimal (Approach 3)
N_opt = (C / (2 * 6))^0.5   # simplified; use optimizer for exact
D_opt = 20 * N_opt           # 20 tokens per parameter

# 6ND FLOP estimate
C = 6 * N * D                # forward + backward

# A100 compute budget
C_total = 312e12 * 0.35 * T_seconds   # at 35% MFU
```

---

## Key Code Pattern — Gate Self-Check

```python
# Quick gate verification snippets

# Criterion 1: count params
def count_params(model):
    return sum(p.numel() for p in model.parameters())
# Should print 50_000_000 to 65_000_000

# Criterion 2: perplexity
import torch, math
def compute_perplexity(model, val_loader, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            ids = batch["input_ids"].to(device)
            out = model(ids, labels=ids)
            total_loss += out.loss.item() * ids.numel()
            total_tokens += ids.numel()
    return math.exp(total_loss / total_tokens)

# Criterion 3: dataset validity check
from datasets import load_dataset
import sqlglot
ds = load_dataset("your-hf-username/postgres-sql-v1")
def is_valid(ex):
    try:
        sqlglot.parse(ex["sql"], dialect="postgres")
        return True
    except:
        return False
valid_rate = sum(is_valid(e) for e in ds["train"]) / len(ds["train"])
print(f"SQL validity: {valid_rate:.1%}")  # must be >= 0.98
```

---

## Decision Rules

- If any criterion is FAIL: spend 1 full week remediating it before starting Phase 4.
- If 1–2 criteria are CONDITIONAL: proceed to Phase 4, complete remediation in Weeks 28–29.
- If all PASS: start Phase 4 immediately.
- Never fine-tune your 50M toy model on SQL data — it is too small. Use Qwen2.5-Coder-7B.
- Perplexity disagreement > 50% between your eval.py and lm-eval-harness = bug in your eval.py.

---

## Numbers to Remember

- 50M model param range: 50–65M is acceptable (56.7M for the reference config)
- Val loss gate: < 4.0 (PPL < 55)
- Dataset gate: ≥5,000 total, ≥4,000 train, ≥1,000 val
- SQL validity gate: ≥98%
- Chinchilla 20-token rule: optimal D = 20N
- Phase 4 base model: Qwen2.5-Coder-7B (7.6B params, Apache 2.0)

---

## Red Flags

- Perplexity is < 10: likely evaluating on training data, not validation data.
- Perplexity is > 100: training did not converge, or tokenizer mismatch.
- SQL validity < 95%: your Tier 3 self-instruct generation needs stricter filtering.
- Dataset on Hub but fields are wrong: re-check ChatML format — must have `system`, `user`, `assistant` keys or a single `messages` list.
- You can explain Chinchilla with notes but not without: you have memorized, not understood. Redo the derivation from scratch.
