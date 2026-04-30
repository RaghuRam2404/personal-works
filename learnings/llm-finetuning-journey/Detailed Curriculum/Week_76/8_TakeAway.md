# Week 76 TakeAway — Multi-Turn Agentic SQL with Tool Use

**This week in 15 words:** Train on error-correction trajectories; run a database-connected loop that self-corrects SQL at inference time.

---

## Key Code Pattern — Loss Masking for Multi-Turn

```python
def mask_non_assistant_tokens(labels, input_ids, assistant_token_ids, eos_token_id):
    """Set labels to -100 for all non-assistant tokens."""
    mask = torch.ones_like(labels) * -100
    i = 0
    while i < len(input_ids):
        if input_ids[i:i+len(assistant_token_ids)].tolist() == assistant_token_ids:
            i += len(assistant_token_ids)
            while i < len(input_ids) and input_ids[i] != eos_token_id:
                mask[i] = labels[i]
                i += 1
        else:
            i += 1
    return mask
```

---

## Key Code Pattern — Minimal Agentic Loop

```python
def run_agentic_loop(model, tokenizer, messages, conn, max_rounds=3):
    for _ in range(max_rounds):
        out = generate(model, tokenizer, messages)
        sql = parse_tool_call(out)       # returns None if no tool call
        if sql is None:
            return out                   # final answer
        result = safe_execute(conn, sql) # always rollback on error
        messages += [{"role": "assistant", "content": out},
                     {"role": "tool", "content": result}]
    return out
```

---

## Decision Rules

If loss is computed on >60% of tokens per batch → masking is broken; fix label array construction.

If model loops to max_rounds on >10% of examples → training data has uniform round count; add zero-correction examples.

If first-attempt EM drops after agentic SFT → include 50% single-shot (no loop) examples in training mix.

If correction rate < 15% → check whether error messages from psycopg2 are being passed intact; truncation or reformatting loses the signal.

If final EM barely exceeds single-shot baseline (<0.5 pp) → agentic loop cost not justified; invest in more SFT data instead.

---

## Numbers to Remember

- Target unmasked token fraction in agentic SFT: 30–50% per batch
- Learning rate for agentic SFT: 1e-4 (lower than standard SFT 2e-4 to preserve single-shot ability)
- Trajectory dataset size for meaningful correction learning: 500–2000 examples
- Max rounds in production loop: 3 (beyond 3, latency cost exceeds accuracy gain in practice)
- Expected correction rate for a well-trained model: 25–45% of first-attempt failures

---

## Red Flags

Model emits row-like text (numbers, pipes) where SQL should appear → tool-result tokens were not masked during training.

Model always stops at exactly round 2 regardless of query complexity → training data had fixed round count; add variety.

`psycopg2` connection in aborted state after every run → forgot `conn.rollback()` after error; add to the exception handler.

Agentic SFT dramatically hurts single-shot performance (>3 pp drop) → training data had no single-round examples; fix the mix.
