# Week 77 Assignment Solutions — Bilingual NL→SQL: English + Tamil

## Task 1 — Tokenizer Analysis Key Snippet

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")

pairs = [
    ("Show total orders by region",
     "பிராந்தியம் வாரியாக மொத்த ஆர்டர்களை காட்டு"),
    ("Find customers with orders above 1000",
     "1000க்கு மேல் ஆர்டர்கள் உள்ள வாடிக்கையாளர்களை கண்டுபிடி"),
]

for en, ta in pairs:
    en_ids = tokenizer.encode(en)
    ta_ids = tokenizer.encode(ta)
    ratio = len(ta_ids) / len(en_ids)
    unk_count = ta_ids.count(tokenizer.unk_token_id)
    print(f"EN: {len(en_ids)} tokens | TA: {len(ta_ids)} tokens | "
          f"Ratio: {ratio:.1f}x | UNK: {unk_count}")
```

**Expected output for Qwen2.5-Coder:** Ratio of 5–8x. Few or no `<unk>` tokens (the tokenizer uses individual Unicode code-points rather than UNK for unseen scripts). This means Tamil is tokenizable but extremely inefficient.

**Context window implication:** If schema = 800 tokens and your model's context = 2048, you have 1248 tokens for the question + system prompt. At 6x inflation, this allows only ~200 Tamil words — sufficient for typical database questions (10–30 words) with margin to spare.

---

## Task 2 — Translation Pipeline Key Snippet

```python
# Using GPT-4o for translation (fastest for a one-week experiment)
import openai, json

client = openai.OpenAI()

def translate_to_tamil(english_question):
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user",
                   "content": (
                       "Translate this database query question to Tamil. "
                       "Keep all SQL column names, table names, numbers, and "
                       "technical terms in English. Return only the Tamil translation.\n\n"
                       f"Question: {english_question}"
                   )}]
    )
    return resp.choices[0].message.content.strip()

# Back-translation check
def back_translate(tamil_text):
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user",
                   "content": f"Translate this Tamil text back to English:\n{tamil_text}"}]
    )
    return resp.choices[0].message.content.strip()
```

**Expected output:** Tamil translations that preserve SQL column and table names in Roman script. Back-translations should recover the original meaning with minor paraphrasing.

---

## Task 3 — Bilingual Training Key Snippet

```python
from datasets import Dataset, concatenate_datasets

# Build bilingual dataset
english_data = Dataset.from_json("sft_train_english.jsonl").select(range(3600))
tamil_data   = Dataset.from_json("week77/tamil_train.jsonl")

# Add bilingual system prompt to all examples
BILINGUAL_SYSTEM = ("You are a PostgreSQL SQL generator. "
                    "The user will ask questions in English or Tamil. "
                    "Always output valid PostgreSQL SQL only.")

def add_bilingual_system(example):
    if example["messages"][0]["role"] == "system":
        example["messages"][0]["content"] = BILINGUAL_SYSTEM
    else:
        example["messages"].insert(0, {"role": "system", "content": BILINGUAL_SYSTEM})
    return example

english_data = english_data.map(add_bilingual_system)
tamil_data   = tamil_data.map(add_bilingual_system)

combined = concatenate_datasets([english_data, tamil_data]).shuffle(seed=42)
```

---

## Common Gotchas

- **Back-translation as a quality gate:** If the back-translation changes the meaning ("total orders" → "order count" is fine; "total orders" → "customer list" is not), flag and discard that example.
- **Tamil SQL column names:** Translators sometimes translate column names (e.g., "region" → "பகுதி"). Your prompt must explicitly instruct the translator to keep column and table names in English. Re-check every translated example for this.
- **Tamil validation loss plateauing high:** With 400 Tamil examples and an English-centric tokenizer, Tamil validation loss will plateau 20–40% higher than English loss. This is expected — do not reduce LR in response.
- **Language interference signal:** If English validation loss rises by more than 0.05 nats after adding Tamil, your mixing ratio is too Tamil-heavy. Reduce to 5% Tamil and re-run.

---

## How to Verify You Did It Right

1. Pick 5 Tamil test questions from `custom200_tamil.jsonl`. Run the model. Verify the output is valid SQL (not Tamil text, not English rephrasing of the question).
2. Check that English EM on Custom-200 after bilingual training is within 1 pp of the pre-training English EM. A larger drop means language interference — adjust the mixing ratio.
3. Open W&B and confirm two separate validation curves are logged: `val_loss_english` and `val_loss_tamil`. They should diverge in the expected direction (Tamil higher).
4. Run `tokenizer_analysis.py` on 5 Tamil evaluation examples. Confirm none exceed your context budget calculation from Task 1.
