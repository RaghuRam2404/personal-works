# Week 77 Curriculum — Bilingual NL→SQL: English + Tamil

## Learning Objectives

By the end of this week, you will be able to:

- Assess the tokenizer coverage of your base model for Tamil script and identify the practical consequences of poor coverage.
- Build a parallel bilingual NL→SQL dataset with Tamil question variants for your existing training examples.
- Fine-tune your model on bilingual data with a balanced mixing strategy that preserves English performance while adding Tamil.
- Evaluate your model across both languages on Custom-200 and diagnose per-language failure patterns.
- Articulate the limits of the current approach and what a production-grade bilingual system would require.

---

## Concepts

### Why Bilingual NL→SQL Matters for Your Context

Tamil is one of the world's oldest classical languages with over 70 million native speakers, predominantly in Tamil Nadu (India) and Sri Lanka. As you build tools for Indian enterprises and government systems, the ability to query databases in Tamil is commercially and socially significant. The standard NL→SQL pipeline assumes English input; extending it to Tamil creates three distinct challenges: tokenizer coverage, training data availability, and cross-lingual generalization.

This week is an honest engineering exploration, not a claim that you will achieve production-grade Tamil NL→SQL in one week. The goal is to understand the constraints, implement a reasonable baseline, measure where it falls short, and document the gap clearly.

### Tokenizer Coverage for Tamil

Every modern LLM tokenizer (BPE, SentencePiece, tiktoken) is built from a vocabulary derived from the pretraining corpus. Models pretrained heavily on English (and code) have tokenizers that represent English and code very efficiently — most common English words are single tokens. Tamil, as a Dravidian language with agglutinative morphology, has fundamentally different structure: a single Tamil word often encodes what would be a clause in English, and the script uses Unicode characters in the range U+0B80–U+0BFF.

The practical consequence of poor Tamil tokenizer coverage is token explosion. A Tamil word that should be one or two tokens may tokenize into 8–12 individual Unicode code-point tokens. This does three things: (1) it compresses the effective context window — a Tamil question that is 10 words long may consume 80+ tokens; (2) it increases computational cost per token; (3) it means the model has seen very few coherent Tamil word sequences during pretraining and must learn Tamil semantics almost from scratch during fine-tuning.

You can measure tokenizer coverage with a simple test:

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")

english = "Show me total orders by region"
tamil   = "பிராந்தியம் வாரியாக மொத்த ஆர்டர்களை காட்டு"

print(len(tokenizer.encode(english)))  # expect 7-9 tokens
print(len(tokenizer.encode(tamil)))    # expect 40-80 tokens
```

For Qwen2.5-Coder, expect 5–8x token inflation for Tamil text. This is a hard constraint on your approach.

### Training Data for Tamil NL→SQL

There is no large-scale Tamil NL→SQL dataset equivalent to Spider or BIRD-SQL. You have two practical sources:

**Translation-based augmentation.** Take your existing English NL→SQL training set (25K+ examples) and translate the natural language questions to Tamil using a translation API or model (IndicTrans2, Google Translate, or GPT-4o). The SQL targets remain identical — SQL is universal. This is fast but introduces translation noise: some Tamil translations will be unnatural or incorrect, particularly for technical database terms (column names, function names) that translators may handle inconsistently.

**Manual templates.** For your most common query patterns (5–10 templates), write Tamil question variants by hand or with a fluent Tamil speaker. This is higher quality but covers fewer examples. For a one-week experiment, targeting 200–500 Tamil examples is realistic.

For this week, target 300–500 Tamil training examples as a pilot. This is not enough to achieve high Tamil accuracy, but it is enough to observe whether the model can learn to map Tamil questions to SQL at all, and to measure the performance gap.

### Bilingual Training Strategy

The central challenge in bilingual fine-tuning is language interference: adding Tamil examples can degrade English performance if the training mix is poorly designed.

**Mixing ratio.** For a dominant English system with Tamil as secondary, start with 90% English, 10% Tamil. This preserves the weight space learned for English while introducing Tamil signal. If you have 5000 total training examples, use 4500 English and 500 Tamil.

**Language tags.** Explicitly mark the language in the system prompt or user turn:

```
System: You are a PostgreSQL SQL generator. The user will ask questions in English or Tamil. Always output valid PostgreSQL SQL.
User (Tamil): பிராந்தியம் வாரியாக மொத்த ஆர்டர்களை காட்டு
```

Language tags help the model disambiguate input language without relying solely on script detection from context.

**Separate vs. mixed batches.** Mixing Tamil and English examples within the same batch is fine for standard SFT. Some practitioners prefer separate Tamil-only batches to avoid gradient conflicts, but there is no strong empirical evidence this is necessary at the scale you are working at (sub-1000 Tamil examples).

### Cross-Lingual Generalization and Its Limits

Why would a model trained on English NL→SQL transfer at all to Tamil? The key insight is that the SQL targets are shared: both `"Show me total orders by region"` and `"பிராந்தியம் வாரியாக மொத்த ஆர்டர்களை காட்டு"` map to the same SQL. During training, the model sees Tamil input → same SQL output. Even with imperfect Tamil representations (high token inflation), the shared SQL target creates a weak cross-lingual bridge.

This bridge is real but limited. It works for query patterns seen during training. For novel Tamil phrasings, the model has no cross-lingual transfer mechanism beyond what was encoded in pretraining (which is minimal for code-focused models). This is why multilingual models trained on diverse languages from scratch (like mT5, BLOOM, or IndicBERT) dramatically outperform code-specialized models on low-resource language tasks.

The honest conclusion: your one-week bilingual experiment will demonstrate proof-of-concept Tamil NL→SQL on patterns seen in training, with significantly lower accuracy than English. It is a foundation for a proper multilingual system that would require: (a) a multilingual base model, (b) 5,000+ Tamil NL→SQL examples, and (c) tokenizer vocabulary extension with Tamil script tokens.

### Evaluation Design

Run Custom-200 in two configurations:

1. **English-only evaluation:** Your 200 test examples with English questions (baseline: 83.1% EM).
2. **Tamil-translated evaluation:** Translate the 200 Custom-200 questions to Tamil, run the same model, compute EM against the same SQL targets.

Report both. Expect Tamil EM to be substantially lower — possibly 40–65% for common patterns, near zero for complex or rare patterns. This gap is the honest measurement of what this approach can and cannot do.

---

## Connections

This week connects to Week 71–74 (frontier reading on multilingual models), Week 53 (dataset construction) for the parallel data pipeline, and Week 65–66 (deployment) because a bilingual production system requires language detection at the API layer. The tokenizer coverage analysis links back to Week 63–64 (quantization and tokenization effects). Week 78 will include your bilingual results as part of the final capstone assessment.

---

## Common Misconceptions / Pitfalls

The SQL output does not need to be translated — only the input question. This is the key simplification that makes bilingual NL→SQL tractable: the label space is the same for both languages.

Translation quality matters more than quantity. 200 high-quality Tamil examples outperform 2000 noisy machine-translated ones for a one-week experiment. Spot-check translations manually.

Do not expect Tamil performance to match English. The goal is a working bilingual prototype, not parity. Be explicit about this in your results memo.

High token inflation for Tamil is not a bug you can fix — it is a property of the tokenizer. The only structural fix is vocabulary extension (adding Tamil characters as atomic tokens), which requires re-embedding initialization and is a research project, not a one-week task.

---

## Time Allocation (6–8 hours)

- 1 hour: Tokenizer coverage analysis; measure token inflation for Tamil vs English; document findings.
- 1.5 hours: Build Tamil training set (translate 300 English questions, spot-check 30 manually).
- 1.5 hours: Fine-tune on bilingual mix (90/10 English/Tamil, 1000 steps); log to W&B `week-77-bilingual-sql`.
- 1.5 hours: Evaluate on Custom-200 in both English and Tamil; compute per-language EM; diagnose failure patterns.
- 0.5 hours: Write results memo documenting performance gap and production requirements.
