# Week 22 — Evaluate the 50M LM: Perplexity, Samples, and Writeup

## Learning Objectives

By the end of this week, you will be able to:

- Compute validation perplexity correctly using log-likelihood accumulation
- Generate text samples with temperature, top-k, and top-p sampling
- Critically analyze generated samples to identify failure modes
- Write a structured model evaluation report
- Recognize what poor perplexity means and diagnose causes

---

## Concepts

### Computing Validation Perplexity Correctly

Perplexity is the exponential of the average negative log-likelihood per token:

```
Perplexity = exp(- (1/N) × Σ log P(x_i | x_<i))
           = exp(cross_entropy_loss)
```

For your 50M model, perplexity should be computed on held-out data (your `val.bin`) that the model never trained on.

**Correct implementation:**

```python
@torch.no_grad()
def compute_perplexity(model, val_data, block_size, batch_size=16, device='cuda'):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n_batches = min(len(val_data) // block_size, 200)  # cap at 200 batches

    for i in range(0, n_batches * block_size, block_size):
        x = val_data[i:i+block_size].unsqueeze(0).to(device)
        y = val_data[i+1:i+block_size+1].unsqueeze(0).to(device)
        logits, loss = model(x, y)
        total_loss += loss.item() * block_size
        total_tokens += block_size

    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)
```

**Critical detail:** Do NOT average the per-batch losses directly without weighting by token count. If your last batch is shorter, it must be handled or excluded. The safest approach is to use non-overlapping windows of exactly `block_size` tokens.

**Sliding window perplexity:** A more accurate approach computes perplexity with a stride shorter than context_len, so each token is predicted with maximum context. For evaluation on your small model this is overkill — fixed stride is fine.

### What Your Perplexity Should Be

| Model | Params | Tokens | Expected PPL on web text |
|---|---|---|---|
| Your 50M model | 56M | ~2B | 25–40 |
| GPT-2 small | 117M | 40B | ~18 |
| GPT-2 medium | 345M | 40B | ~14 |
| GPT-Neo 125M | 125M | ~300B | ~12 |

If your perplexity is above 50: likely causes include insufficient training, broken data pipeline, or tokenizer mismatch.
If your perplexity is below 15 on FineWeb val: likely data contamination (val set overlaps train set).

### Text Generation: Sampling Methods

Once you have the model, you can generate text by autoregressively sampling one token at a time.

**Greedy decoding:** Always pick the highest-probability token. Deterministic. Produces repetitive, degenerate text for open-ended generation.

**Temperature sampling:** Divide logits by temperature T before softmax:
- T < 1.0: sharpens the distribution → more predictable, less diverse
- T = 1.0: exact model distribution
- T > 1.0: flattens the distribution → more random, less coherent

```python
def generate(model, prompt_ids, max_new_tokens=100, temperature=0.8, top_k=50):
    model.eval()
    x = prompt_ids.unsqueeze(0)  # (1, T)
    for _ in range(max_new_tokens):
        x_cond = x[:, -context_len:]  # truncate to context window
        with torch.no_grad():
            logits, _ = model(x_cond)
        logits = logits[:, -1, :] / temperature   # (1, vocab_size)
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, -1:]] = -float('inf')
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_token], dim=1)
    return x[0, len(prompt_ids):]  # return only new tokens
```

**Top-k sampling:** Only sample from the top-k most probable tokens. Prevents sampling very improbable tokens that produce garbage.

**Top-p (nucleus) sampling:** Sample from the smallest set of tokens whose cumulative probability exceeds p. Adapts the candidate set to the model's confidence at each step.

**Recommended for your evaluation:** temperature=0.8, top_k=50. This gives readable but diverse samples.

### Analyzing Generated Samples

Generate at least 10 samples with different prompts. Evaluate each on:

1. **Fluency:** Is the text grammatically correct English?
2. **Coherence:** Does the text stay on topic for more than 2–3 sentences?
3. **Repetition:** Does the model get stuck in loops ("the the the the")?
4. **Factual accuracy:** Does the model generate plausible facts (not required to be true, but should not be incoherent)?
5. **Domain:** Does the model "know" anything about databases, SQL, or programming?

At 50M parameters trained on general web text, you should expect:
- Mostly fluent English for short spans (5–10 words)
- Loss of coherence after 30–50 tokens
- Some repetition, especially with greedy or low-temperature sampling
- No meaningful SQL or technical knowledge (the model is too small and general)

This is the expected result. Document it honestly.

### What to Include in Your Writeup

Your writeup (`week-22-evaluation.md`) should have these sections:

1. **Training Summary:** Total tokens, final val loss, best val loss, training hours, checkpoint link
2. **Perplexity Analysis:** Val perplexity on FineWeb val. Baseline: GPT-2-small perplexity for comparison.
3. **Generated Samples:** 5–10 samples with prompts, temperature, and top_k settings. Annotate each.
4. **Failure Analysis:** What does your model do badly? Be specific.
5. **Scaling Takeaways:** What would improve this model most — more parameters, more data, better architecture, or better data quality?
6. **Lessons for Fine-Tuning:** Given this base model, what would fine-tuning on PostgreSQL SQL data actually improve?

Section 5 and 6 are the most important. This is the transfer from pretraining to fine-tuning thinking.

### Diagnosing High Perplexity

If your perplexity is higher than expected (>40 on FineWeb val):

| Cause | Diagnostic | Fix |
|---|---|---|
| Insufficient training | Check tokens_seen < 500M | Train longer |
| Wrong val set | Check val.bin was not seen during training | Rebuild from different shard |
| Tokenizer mismatch | Encode/decode a sentence, check roundtrip | Retrain tokenizer with same settings |
| Model architecture bug | Check weight tying, pre-LN order | Audit model.py against curriculum |
| Bad data quality | Sample 10 documents from train.bin | Re-filter dataset |

---

## Connections

**Prior weeks (20–21):** Your checkpoint and data pipeline from those weeks drive this evaluation.

**Week 23:** The evaluation methods here (perplexity computation, sampling) are a subset of what lm-evaluation-harness automates. This week gives you the manual foundation.

**Phases 4–6:** The failure analysis from this week directly informs why you will choose a pretrained 7B SOTA model for fine-tuning instead of your 50M model.

---

## Common Misconceptions

- **"Low training loss = good model."** Training loss measures in-distribution performance. A model can have low training loss but high val loss (overfitting).
- **"Perplexity is comparable across different tokenizers."** Perplexity is tokenizer-dependent. GPT-2's perplexity on a 50K-vocab tokenizer is not directly comparable to your perplexity on a 32K-vocab tokenizer without adjustment.
- **"Temperature 1.0 is always best."** Temperature 1.0 gives exact model probabilities, but for generation tasks with open-ended prompts, 0.7–0.9 often produces more coherent output by reducing extreme low-probability samples.
- **"Repetition means the model is broken."** Repetition is a known failure mode of small LMs and even large ones without repetition penalties. It reflects the n-gram language model bias toward high-frequency patterns.

---

## Time Allocation (6–8 hrs)

- 1h: Implement and validate perplexity computation
- 1h: Implement text generation with temperature + top-k sampling
- 1.5h: Generate 15–20 samples across various prompts and temperatures
- 2h: Write the evaluation report (`week-22-evaluation.md`)
- 0.5h: Upload checkpoint to HuggingFace (if not done in Week 21)
- 0.5h: Commit and write journal entry
