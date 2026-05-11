# Week 23 TakeAway — LM Evaluation

**One-liner:** lm-eval uses log-likelihood to score multiple-choice; MMLU tests facts; HellaSwag tests commonsense; your SQL model needs execution accuracy, not MMLU.

---

## Key Command

```bash
# Install
pip install lm-eval

# Evaluate a HuggingFace model
lm_eval \
  --model hf \
  --model_args pretrained=gpt2 \
  --tasks hellaswag,arc_easy,arc_challenge \
  --device cuda:0 \
  --batch_size 16 \
  --output_path results/gpt2.json
```

---

## Key Code Pattern — Log-Likelihood Scoring

```python
def score_option(model, context_ids, option_ids, device):
    """Score an option given a context using per-token log-likelihood."""
    x = torch.tensor(context_ids + option_ids[:-1]).unsqueeze(0).to(device)
    y = torch.tensor(context_ids + option_ids)[1:].unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _ = model(x)
    log_probs = F.log_softmax(logits, dim=-1)
    option_log_prob = log_probs[0, len(context_ids)-1:, :].gather(
        1, y[0, len(context_ids)-1:].unsqueeze(1)
    ).sum()
    return option_log_prob.item() / len(option_ids)  # length-normalized
```

---

## Benchmark Cheat Sheet

| Benchmark | Type | Random | GPT-2 (117M) | GPT-4 |
|---|---|---|---|---|
| HellaSwag | Commonsense MC | 25% | 31.6% | ~95% |
| ARC-Easy | Science MC | 25% | 43.2% | ~96% |
| ARC-Challenge | Hard science MC | 25% | 25.9% | ~96% |
| MMLU | 57-topic facts | 25% | ~26% | ~87% |
| GSM8K | Math word probs | ~0% | ~0% | ~92% |

Expected for your 50M model: 26–35% on all MC tasks (near random to slightly above).

---

## Decision Rules

- Use lm-eval log-likelihood scoring for any multiple-choice benchmark → no instruction tuning needed
- 0-shot vs 5-shot: always report which one; 5-shot is typically higher
- Model scores below random (< 25% on 4-choice) → sign error in log-likelihood computation
- For SQL model evaluation → use Execution Accuracy, not MMLU/HellaSwag
- Benchmark contamination suspected → compare model accuracy on public test vs. private held-out set

---

## Numbers to Remember

| Fact | Value |
|---|---|
| Random baseline (4-choice) | 25% |
| GPT-2 HellaSwag | 31.6% |
| GPT-2 ARC-Easy | 43.2% |
| GPT-4 MMLU | ~87% |
| Human MMLU | ~89% |
| 50M model MMLU | ~25-27% |

---

## Red Flags

- Your model scores 100% on MMLU → data contamination
- Your model scores < 20% on any 4-choice benchmark → log-likelihood sign error
- Perplexity is good but all downstream tasks score near random → distribution mismatch or insufficient capacity
