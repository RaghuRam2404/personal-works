# Week 77 TakeAway — Bilingual NL→SQL: English + Tamil

**This week in 15 words:** Tamil NL→SQL is feasible as a prototype but requires honest acknowledgment of tokenizer and data constraints.

---

## Key Code Pattern — Token Inflation Measurement

```python
def token_inflation(tokenizer, en_text, ta_text):
    en_count = len(tokenizer.encode(en_text))
    ta_count = len(tokenizer.encode(ta_text))
    return ta_count / en_count, en_count, ta_count

ratio, en_n, ta_n = token_inflation(tokenizer, "Show orders by region",
                                    "பிராந்தியம் வாரியாக ஆர்டர்களை காட்டு")
print(f"Inflation: {ratio:.1f}x ({en_n} → {ta_n} tokens)")
```

---

## Key Code Pattern — Bilingual Dataset Mix

```python
from datasets import concatenate_datasets

# 90/10 English/Tamil mix
combined = concatenate_datasets([
    english_dataset.select(range(3600)),
    tamil_dataset.select(range(400))
]).shuffle(seed=42)

# Bilingual system prompt added to ALL examples (not just Tamil)
SYSTEM = ("You are a PostgreSQL SQL generator. "
          "Accept questions in English or Tamil. "
          "Output valid PostgreSQL SQL only.")
```

---

## Key Code Pattern — Back-Translation Quality Gate

```python
def is_translation_valid(original_en, tamil_text, back_en, threshold=0.6):
    """Simple lexical overlap check — not perfect but catches major errors."""
    orig_words = set(original_en.lower().split())
    back_words = set(back_en.lower().split())
    overlap = len(orig_words & back_words) / max(len(orig_words), 1)
    return overlap >= threshold
```

---

## Decision Rules

If token inflation > 8x → evaluate whether your context budget still covers 40-word Tamil questions; truncate schema if needed.

If English EM drops > 1 pp after bilingual training → first check if the bilingual system prompt (not training data) is causing the drop; test prompt on pre-training checkpoint.

If Tamil EM < 40% → more data before changing the model; poor performance at this level is a data problem, not a model problem.

If Tamil EM plateau above 60% requires breaching → switch to a multilingual base model or extend tokenizer vocabulary.

If Tamil column-name translation errors appear in output → add explicit instruction in system prompt: "All column and table names must appear in English."

---

## Numbers to Remember

- Typical Tamil token inflation on Qwen2.5-Coder: 5–8x
- Training mix for safe bilingual fine-tuning: 90% English / 10% Tamil
- Tamil training examples for a first prototype: 300–500
- Tamil training examples to reach ~75% EM: estimated 5,000–15,000
- Expected prototype Tamil EM: 40–60% (vs English 83.1%)

---

## Red Flags

Tamil output appears in place of SQL → model received a Tamil-language training example where the label was also in Tamil (translation error); audit your dataset labels.

English EM drops sharply after 200 Tamil steps → mixing ratio too high; reduce Tamil proportion.

Tamil validation loss never decreases → translation quality is too noisy; the model cannot extract consistent signal; review translations manually.

All Tamil test queries produce identical SQL → model collapsed to a high-frequency template; training data lacks diversity in Tamil examples.
