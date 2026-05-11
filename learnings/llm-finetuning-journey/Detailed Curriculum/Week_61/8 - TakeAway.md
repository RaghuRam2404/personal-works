# Week 61 TakeAway — Evaluation Harness

**One-liner:** Always use dev set for selection, execution accuracy over result comparison, and report confidence intervals.

---

## Eval Harness CLI Pattern

```python
# eval_harness.py --model <path> --benchmark bird --split dev --output results.json
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--benchmark", choices=["bird","spider","custom"])
parser.add_argument("--max-examples", type=int, default=None)
parser.add_argument("--timeout", type=int, default=30)
parser.add_argument("--output", default="eval_results.json")
```

## Batch Inference (Speed)

```python
@torch.no_grad()
def batch_generate(prompts, model, tokenizer, batch_size=8):
    # Pad left for decoder-only models
    tokenizer.padding_side = "left"
    for batch in chunks(prompts, batch_size):
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=2048).to(device)
        out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        yield [decode_new_tokens(o, inputs) for o in out]
```

## Result Comparison

```python
def results_match(pred_rows, ref_rows):
    if pred_rows is None: return False
    # Sort rows as strings; handles different column orders
    return (sorted(str(r) for r in pred_rows) == 
            sorted(str(r) for r in ref_rows))
```

---

## Decision Rules

- NEVER select the final model based on test set performance — use dev set only
- Use greedy decoding (do_sample=False) for eval — reproducible and deterministic
- Set 30-second timeout per query — runaway queries count as failures
- Report CI alongside point estimates (bootstrap with 1,000 samples)
- Compare on your CUSTOM benchmark first — it's your honest domain measure

---

## Numbers to Remember

- BIRD-SQL dev: 1,534 questions ("Simple"/"Moderate"/"Challenging" labels)
- Spider 1.0 dev: 1,034 questions (cross-database, SQLite-native)
- Defog sql-eval: ~200 enterprise SQL questions
- Typical SOTA on BIRD dev (as of 2024): 65–70% execution accuracy
- Expected range for your model on BIRD dev: 55–70%
- Bootstrap CI: 1,000 resamples, report [2.5%, 97.5%] percentiles

---

## Red Flags

- Eval accuracy is identical for all models: harness bug (not connecting to right DB or not comparing correctly)
- Spider accuracy > BIRD accuracy by < 5pp: Spider is notably easier — check your harness
- All failures are timeout: your eval DB has missing indexes — add indexes to speed up reference queries
- Custom benchmark accuracy not broken down by skill: insufficient for technical report
