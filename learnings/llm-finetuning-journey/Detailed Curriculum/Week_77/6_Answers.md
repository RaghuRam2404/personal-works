# Week 77 Answers — Bilingual NL→SQL: English + Tamil

---

## Q1. Answer: D

**Why D is correct:** 25 Tamil words × 7 tokens/word = 175 tokens for the question. Remaining budget after schema = 2048 - 900 = 1148 tokens. 175 < 1148, so it fits. The calculation is straightforward but the key insight is to frame it as available budget, not total context. Simple Tamil database questions (10–30 words) will always fit even with 7x inflation; the risk emerges with complex multi-clause questions (50+ words) or when the schema is large.

**Why A is wrong:** 175 tokens is correct, but "comfortably" is misleading. With a 40-word Tamil question, you would consume 280 tokens — still fine, but the margin shrinks and the framing should reflect that.

**Why B is wrong:** Tamil tokenizes at 5–8x the rate of English for code-specialized tokenizers. Equivalence is false.

**Why C is wrong:** 1148 tokens is the budget (2048 - 900), not the question size. This conflates question cost with available space.

---

## Q2. Answer: C

**Why C is correct:** The bilingual system prompt was added to all training examples, but your pre-training English evaluation used the original single-language prompt. When you evaluate English EM post-training with the new bilingual system prompt, the model is in a different prompt distribution. English EM may drop 1–2 pp purely due to this prompt shift, not because the model forgot English. To verify: run English evaluation with both the old prompt and the new bilingual prompt on the pre-training checkpoint — if the bilingual prompt alone reduces EM by ~2 pp, the observed drop after bilingual training is a prompt artifact.

**Why A is wrong:** Models do not forget vocabulary from seeing new tokens — the embedding space is additive, not overwriting. Vocabulary forgetting is a misconception.

**Why B is wrong:** Tamil and English word order differences are not a source of gradient conflict in fine-tuning — the model operates on token representations, not syntactic trees.

**Why D is wrong:** More training steps could help Tamil, but they are not the explanation for the English drop. English forgetting from over-training would manifest as increased English validation loss throughout training, which you can check in W&B.

---

## Q3. Answer: B

**Why B is correct:** Starting with a very high English ratio (99%) anchors the model's weight space in the English performance regime. Gradually increasing Tamil allows the model to pick up Tamil signal without disrupting the English gradient landscape. This is an application of curriculum learning — easy (well-represented) before hard (under-represented). Sequential training (A) risks catastrophic forgetting of Tamil in the final English stage.

**Why A is wrong:** Sequential training (Tamil then English) will cause the model to partially forget Tamil during the final English fine-tuning stage, which is exactly what you are trying to avoid.

**Why C is wrong:** 50/50 mixing gives equal weight to 400 Tamil examples and 400 English examples from each batch perspective. This severely under-represents English relative to its importance and risks significant English degradation.

**Why D is wrong:** This is not how DPO works with language data. DPO uses preference pairs, not language pairs as rejected examples. Tamil output is not a "rejected" response — it is simply a response to a Tamil input, and the SQL target is correct regardless of language.

---

## Q4. Answer: A

**Why A is correct:** Vocabulary extension addresses the root cause of the 7x inflation problem. When Tamil characters or common Tamil morphemes are represented as atomic tokens (rather than sequences of Unicode code-point tokens), each Tamil word maps to 1–3 tokens instead of 8–12. This restores the effective context window, reduces computational cost, and — most importantly — means the model processes Tamil as coherent units rather than fragmented code-points, improving its ability to learn semantic meaning from Tamil training data.

**Why B is wrong:** BLEU on translation is irrelevant here — you are generating SQL, not translating Tamil to English.

**Why C is wrong:** Tamil and English phonemes are fundamentally different (Dravidian vs Indo-European). Sharing n-gram tokens across the two languages based on phoneme similarity is not a well-grounded approach and would not create useful cross-lingual representations.

**Why D is wrong:** ISO 639-1 compliance is a language code standard, not a property conferred by vocabulary extension. It is irrelevant to model capability.

---

## Q5. Answer: B

**Why B is correct:** This answer provides the correct technical framing: 51% EM is not a failure — it is a reasonable baseline given specific, quantifiable constraints. The three constraints you can cite are: (1) tokenizer inefficiency (7x inflation), (2) minimal Tamil pretraining data in a code-specialized model, (3) only 400 Tamil training examples. 51% EM means the model correctly generates SQL for more than half of Tamil queries it has never seen before, which demonstrates genuine cross-lingual transfer. The honest addendum is that 51% is not production-ready and closing the gap requires addressing these constraints systematically.

**Why A is wrong:** There is no formal basis for a "difficulty adjustment" between languages in EM. EM is EM — query either matches or it does not.

**Why C is wrong:** While translation errors are a real concern and should be measured, using them to explain away the result without evidence is not honest analysis.

**Why D is wrong:** Exceeding random baseline (0%) is a very low bar. This justification would apply to any above-chance performance and does not constitute a meaningful technical argument.

---

## Q6 — Short Answer

Cross-lingual transfer in NL→SQL means that a model trained on English question → SQL pairs can generalize to Tamil question → SQL pairs, even without explicit Tamil training. It is possible because the SQL output is language-agnostic — `SELECT region, SUM(amount) FROM orders GROUP BY region` is the correct answer regardless of whether the question was asked in English or Tamil. During training on Tamil examples, the model maps Tamil token sequences to the same SQL targets it already knows how to generate from English. The shared structure that enables transfer is the SQL label space: both languages converge on the same logical query structure. Additionally, modern LLMs have some multilingual capacity from pretraining on diverse text (including Wikipedia in many languages), which means Tamil question tokens are not completely opaque — the model has weak but nonzero representations of common Tamil words. This weak cross-lingual bridge, combined with shared SQL targets, enables transfer even from a small Tamil training set.

---

## Q7 — Short Answer

The pattern (68% EM on simple queries, 12% on TimescaleDB functions) reflects two separate problems. For simple queries, cross-lingual transfer works because the mapping from Tamil tokens to SQL primitives (SELECT, WHERE, =, ORDER BY) is learnable from 400 examples. For TimescaleDB-specific functions (time_bucket, first, last), the English model's capability rests on recognizing specific English phrases ("bucket by hour," "first value," "rolling average") and mapping them to function calls. The Tamil translations of these phrases are rare or absent from your 400-example training set, so the model has never seen the Tamil→time_bucket mapping.

Training-data intervention: explicitly add 50–100 Tamil training examples that specifically cover time_bucket, first, and last queries. These should be manually written or carefully translated and spot-checked. Targeted augmentation on the tail of the capability distribution is more efficient than adding more general examples.

Inference-time intervention: implement a query classifier that detects whether a question involves time-series operations (keyword matching on Tamil equivalents of "per hour," "per day," "first value," etc.) and routes those questions to a specialized prompt template that explicitly lists the available TimescaleDB functions with Tamil labels. This is a retrieval-augmented prompt strategy — no model retraining required.

---

## Q8 — Short Answer

Reaching Tamil EM within 5 pp of English EM (~78%) from a current baseline of 51% is a substantial investment. A rough estimate based on NL→SQL transfer learning literature: 5,000–15,000 high-quality Tamil NL→SQL examples would be needed with the current tokenizer. The three main constraints driving this cost are:

First, tokenizer inefficiency: 7x token inflation means each Tamil example consumes 7x more context budget than an English one, reducing effective training signal per compute step. Addressing this via vocabulary extension would reduce the data requirement by an estimated 40–60%.

Second, Tamil NL→SQL data scarcity: no large public Tamil NL→SQL dataset exists. Every example must be created or translated, then verified. Human verification at scale costs money and time.

Third, compositional coverage: SQL query patterns are long-tailed. To reach 78% EM, the model must correctly handle rare query types (multi-level CTEs, window functions, TimescaleDB dialect). Each rare pattern in Tamil requires several training examples to learn. A ballpark: 5K examples covers common patterns; 15K covers the long tail needed for 78%+ EM.

---

## Q9 — Deep Scenario

**Model answer:**

The data pipeline would start by collecting domain-specific Tamil NL→SQL examples from subject-matter experts in agricultural and irrigation databases — not generic translation of English examples, because domain terms (crop yield, water allocation, monsoon period) have culturally specific Tamil phrasing that machine translation handles poorly. Target 3,000 domain-specific Tamil examples, manually verified by a Tamil-speaking database analyst, covering the full schema including weather time-series (time_bucket queries for monsoon data) and spatial joins (district-level aggregations).

For the base model, switch to a multilingual-capable model: Qwen2.5-Coder has minimal Tamil pretraining. A better starting point is a model with documented multilingual pretraining, such as a variant of Aya (Cohere), BLOOM, or — most practically — a model fine-tuned on IndicNLP data. Alternatively, retain Qwen2.5-Coder but extend the tokenizer with 500–1000 Tamil morpheme tokens, initialize their embeddings via aligned translation embeddings, and run continued pretraining on Tamil text for 50M tokens before SFT. This adds computational cost but would allow Tamil EM to scale with data.

The evaluation methodology would track Tamil EM on a domain-specific held-out set (200 Tamil questions, verified by domain experts), stratified by query type (simple filter, aggregation, time-series, spatial join). Weekly evaluation checkpoints ensure Tamil improvement does not degrade English EM beyond 1 pp.

The primary risk specific to Tamil NL→SQL (not present in English) is transliteration ambiguity: some Tamil database terminology is expressed as Tamil-script transliterations of English terms, which varies by region and speaker (e.g., "database" may appear as "தரவுத்தளம்," "டேட்டாபேஸ்," or even the English word). The model must handle all variants. Mitigation: in training data, include examples with each variant spelling; at inference time, add a normalization step that maps transliterated English terms to a canonical form before the model processes the input.
