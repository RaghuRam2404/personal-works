# Week 77 Quiz — Bilingual NL→SQL: English + Tamil

Difficulty: Senior research engineer. Questions require reasoning about multilingual NLP, tokenizer design, cross-lingual transfer, and honest evaluation.

---

## Multiple Choice

**Q1.** Your tokenizer analysis shows that Tamil text inflates token count by 7x relative to English. You have a model with a 2048-token context window. Your schema uses 900 tokens. A Tamil user question is 25 Tamil words. Approximately how many tokens does the question consume, and can it fit?

A. 175 tokens — fits comfortably.
B. 25 tokens — Tamil and English are tokenized equivalently.
C. 1148 tokens — exceeds the remaining 1148-token budget; this is an edge case.
D. ~175 tokens (25 words × 7 tokens/word) — barely fits within 1148 tokens (2048 - 900).

---

**Q2.** You train on 3600 English + 400 Tamil examples (90/10 mix). After training, English EM drops from 83.1% to 80.8% and Tamil EM is 54.2%. The most likely primary cause of the English EM drop is:

A. The model forgot English vocabulary because it saw Tamil tokens during training.
B. The 400 Tamil examples conflicted with English learning because Tamil and English have opposite word orders.
C. The bilingual system prompt changed the prompt distribution during evaluation, and the drop is a prompt sensitivity artifact rather than true capability loss.
D. 1000 training steps were insufficient; the model needs longer training to absorb Tamil without losing English.

---

**Q3.** You want to add Tamil to your system with minimal disruption to English performance. Which strategy is most likely to preserve English EM while establishing Tamil capability?

A. Train exclusively on Tamil data for 500 steps, then fine-tune on English data for another 500 steps (sequential training).
B. Use a 99% English / 1% Tamil mix for the first 800 steps, then increase Tamil ratio to 10% for the final 200 steps.
C. Train on 50% English / 50% Tamil so the model is genuinely bilingual.
D. Train only on English but add Tamil examples during DPO as rejected responses, teaching the model to avoid Tamil output.

---

**Q4.** A colleague proposes vocabulary extension: adding 1000 Tamil character n-grams as new tokens to the tokenizer, initializing their embeddings from the mean of existing similar-sounding phoneme tokens, and then re-training embeddings during fine-tuning. The primary benefit of this approach over your current strategy is:

A. It eliminates the 7x token inflation problem, allowing Tamil questions to be represented as efficiently as English.
B. It improves BLEU score on Tamil-to-English translation, which is a prerequisite for SQL generation.
C. It allows the model to share knowledge across Tamil and English because n-gram tokens capture phoneme-level structure common to both.
D. It makes the model formally multilingual, satisfying ISO 639-1 compliance requirements.

---

**Q5.** You compute Tamil EM on Custom-200 (translated) and get 51%. English EM is 83.1%. A stakeholder says "51% is too low; this is not good enough." You respond that this result is expected given the approach. Which of the following is the best technical justification?

A. Tamil is a harder language than English, so 51% is equivalent to 83% in English difficulty.
B. The base model was pretrained primarily on English and code; the tokenizer has 7x token inflation for Tamil; the training set had only 400 Tamil examples; given these constraints, 51% on novel Tamil queries demonstrates positive cross-lingual transfer and is a reasonable baseline.
C. Custom-200 was designed for English queries; the Tamil translations may have introduced errors, so the true Tamil EM is likely higher than 51%.
D. 51% EM on Tamil exceeds random baseline (~0%), so the model is learning.

---

## Short Answer

**Q6.** Explain what "cross-lingual transfer" means in the context of NL→SQL, and why it is possible even when a base model has never seen Tamil NL→SQL pairs during pretraining. Be specific about what shared structure enables the transfer.

---

**Q7.** You observe that your bilingual model performs well on Tamil questions that map to simple SELECT + WHERE queries (EM = 68%) but very poorly on questions requiring TimescaleDB-specific functions like `time_bucket` (EM = 12%). Diagnose the specific reason for this pattern and propose one training-data intervention and one inference-time intervention to address it.

---

**Q8.** A product manager asks you to estimate the data investment required to achieve Tamil EM within 5 pp of English EM (i.e., around 78%). Based on what you learned this week, what factors determine this estimate? Give a rough quantitative range for the number of Tamil training examples required, and explain the three main constraints that drive the cost.

---

## Deep Scenario

**Q9.** You are designing a production bilingual NL→SQL system for a Tamil Nadu government agency that needs to query a PostgreSQL database containing agricultural yield, weather, and irrigation data. English EM requirement: ≥85%. Tamil EM requirement: ≥75%. Current Tamil EM from your prototype: 51%.

Write a structured engineering plan (5–7 sentences) that covers: (a) the data pipeline you would build to reach the Tamil EM requirement, (b) whether you would use your current base model or switch to a multilingual base model and why, (c) the evaluation methodology to track progress, and (d) one risk that is specific to Tamil NL→SQL that does not exist in English NL→SQL, and how you would mitigate it.
