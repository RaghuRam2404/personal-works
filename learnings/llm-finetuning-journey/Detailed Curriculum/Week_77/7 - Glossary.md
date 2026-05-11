# Week 77 Glossary — Bilingual NL→SQL: English + Tamil

**Token inflation:** The increase in token count when tokenizing a non-English language with an English-dominant tokenizer, expressed as a ratio (Tamil/English).

**Cross-lingual transfer:** A model's ability to apply knowledge learned from one language to tasks in a different language it was not explicitly trained on.

**Agglutinative morphology:** A word-formation system where grammatical meanings are added as suffixes/prefixes; Tamil is agglutinative, meaning one word encodes multi-word English phrases.

**Vocabulary extension:** Adding new tokens to a tokenizer's vocabulary and initializing their embeddings, typically to reduce token inflation for a target language.

**Mixing ratio:** The proportion of training examples from each language in a bilingual dataset; e.g., 90% English / 10% Tamil.

**Back-translation:** Translating a translated text back to the original language as a quality check; significant meaning shift indicates translation error.

**Language interference:** Degradation in performance on a high-resource language (English) caused by training on a low-resource language (Tamil) simultaneously.

**Transliteration ambiguity:** Multiple valid ways to write a foreign-language loanword in a target script (e.g., "database" in Tamil script), creating vocabulary inconsistency in training data.

**Domain-specific translation:** Translation that preserves technical terminology (column names, function names, SQL keywords) exactly while converting the surrounding natural language.

**Prototype EM gap:** The difference in exact match accuracy between the primary language (English) and the secondary language (Tamil) in a bilingual system; expected to be 20–40 pp for a first prototype with limited Tamil data.

**IndicTrans2:** An open-source neural machine translation model developed for Indian languages; preferred over Google Translate for Tamil NL→SQL translation due to domain coverage and open access.

**Multilingual base model:** A pretrained language model trained on data from many languages with deliberate multilingual coverage (e.g., BLOOM, mT5, Aya); preferred over code-specialized models when a secondary language exceeds 30% of queries.
