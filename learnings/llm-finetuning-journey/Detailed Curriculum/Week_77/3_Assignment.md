# Week 77 Assignment — Bilingual NL→SQL: English + Tamil

## Setup Checklist

- [ ] Python environment with `transformers>=4.40`, `trl>=0.8`, `peft>=0.10`, `datasets`
- [ ] Access to a translation tool: IndicTrans2 (preferred, open-source), Google Translate API, or GPT-4o API for Tamil translation
- [ ] Your existing SFT training set (from Week 55 or later) accessible locally
- [ ] Your fine-tuned model checkpoint (`postgres-sqlcoder-7b-final`) available
- [ ] W&B project `week-77-bilingual-sql` created
- [ ] Optional: A Tamil speaker to spot-check 20–30 translations (reach out to a friend or use a Tamil online community forum for quick checks)

---

## Task 1 — Tokenizer Coverage Analysis

**Goal:** Quantify the tokenizer's Tamil coverage and understand its implications for context length and training efficiency.

**Requirements:**
- [ ] Write a script that takes 20 English questions from your training set and their Tamil translations, tokenizes both, and computes: token count (English), token count (Tamil), inflation ratio (Tamil/English), and whether any Tamil tokens map to `<unk>`.
- [ ] Produce a table: `| Question | EN tokens | TA tokens | Ratio |`
- [ ] Compute the mean inflation ratio across your 20 pairs.
- [ ] Determine: given your model's context window (typically 2048 or 4096 tokens), what is the maximum Tamil question length (in Tamil words) that fits alongside a typical schema (800 tokens)?
- [ ] Log findings to `week77/tokenizer_analysis.md`.

**Deliverable:** `week77/tokenizer_analysis.py` + `week77/tokenizer_analysis.md`.

---

## Task 2 — Build a Tamil NL→SQL Dataset

**Goal:** Produce 300–500 Tamil NL→SQL training examples.

**Requirements:**
- [ ] Select 400 examples from your existing English SFT dataset, prioritizing: (a) simple single-table queries (50%), (b) JOIN queries (30%), (c) TimescaleDB time_bucket queries (20%).
- [ ] Translate the natural language question (not the SQL) to Tamil using your chosen translation tool.
- [ ] Spot-check at least 30 translations manually. If you cannot read Tamil, use back-translation (Tamil → English) to verify meaning is preserved. Flag and replace any translation with a meaning shift greater than word-for-word reordering.
- [ ] Format each example in your model's chat template with a bilingual system prompt that says the model accepts questions in English or Tamil.
- [ ] Save to `week77/tamil_train.jsonl` (400 examples) and `week77/tamil_val.jsonl` (50 examples, from a separate held-out set).

**Deliverable:** `week77/build_tamil_dataset.py` + `week77/tamil_train.jsonl` + `week77/tamil_val.jsonl`.

**Hints:** For IndicTrans2: `pip install indictrans2`. The model accepts `src_lang="eng_Latn"` and `tgt_lang="tam_Taml"`. For GPT-4o translation, use a prompt like: `"Translate this database question to Tamil. Keep all SQL column names, table names, and numbers in English. Return only the Tamil translation."`

---

## Task 3 — Bilingual Fine-Tuning

**Goal:** Fine-tune your model on a bilingual mix (90% English, 10% Tamil) and measure whether Tamil capability is acquired without hurting English.

**Requirements:**
- [ ] Combine 3600 English SFT examples + 400 Tamil examples = 4000 total.
- [ ] Shuffle the combined dataset uniformly (do not sort by language — interleave Tamil and English examples across batches).
- [ ] Add the bilingual system prompt to all examples (both English and Tamil), not just Tamil examples, so the model is always in "bilingual mode."
- [ ] Train for 1000 steps with LoRA (r=64, alpha=128, lr=2e-4).
- [ ] Log to W&B `week-77-bilingual-sql`: training loss, Tamil validation loss (computed on `tamil_val.jsonl` only), English validation loss (computed on a 200-example English held-out set).
- [ ] Checkpoint at step 500 and step 1000.

**Deliverable:** `week77/train_bilingual.py` + W&B run link + checkpoints.

---

## Task 4 — Bilingual Evaluation on Custom-200

**Goal:** Measure per-language EM on your benchmark and document the performance gap.

**Requirements:**
- [ ] Translate all 200 Custom-200 questions to Tamil (use the same tool as Task 2 for consistency). Save to `week77/custom200_tamil.jsonl`.
- [ ] Run evaluation: English Custom-200 → EM. Tamil Custom-200 → EM. Use the same model checkpoint (step 1000).
- [ ] Stratify Tamil results: (a) simple single-table EM, (b) JOIN EM, (c) TimescaleDB EM. Report all three.
- [ ] Compare English EM before and after bilingual training (did it drop?). A drop of >1 pp is a warning sign of language interference.
- [ ] Write `week77/results_memo.md` covering: current Tamil EM, gap to English EM, top 3 failure patterns for Tamil, what would be needed for production-grade Tamil accuracy (specific data quantity, tokenizer change, or model change).

**Deliverable:** `week77/eval_bilingual.py` + `week77/results_memo.md` + `week77/bilingual_results.json`.

---

## Stretch Goals

- Implement a language detection layer: a simple classifier (logistic regression on character n-grams, or a call to `langdetect` library) that routes Tamil questions to a Tamil-specific prompt template and English questions to the standard prompt. Measure if explicit routing improves Tamil EM by 2+ pp.
- Experiment with 80/20 and 70/30 English/Tamil mixing ratios. Plot Tamil EM vs English EM as a function of mixing ratio. Find the Pareto-optimal ratio for your use case.
- Investigate whether IndicBERT or BLOOM embeddings could be used as a cross-lingual translation layer feeding into your SQL generator (research direction, no code required — write a 300-word design sketch in `week77/multilingual_design.md`).
