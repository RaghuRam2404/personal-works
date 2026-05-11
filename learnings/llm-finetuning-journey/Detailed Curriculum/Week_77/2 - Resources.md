# Week 77 Resources — Bilingual NL→SQL: English + Tamil

## Papers

[Dr.Spider: A Diagnostic Evaluation Benchmark towards Text-to-SQL Robustness](https://arxiv.org/abs/2301.08881) — Chang et al., ICLR 2023. 17 perturbations on Spider that quantify how badly models break under question paraphrases and schema noise — the exact failure modes you should expect when crossing into Tamil. Use it as your robustness baseline before evaluating Tamil performance.

[IndicTrans2: Towards High-Quality and Accessible Machine Translation for All 22 Scheduled Indian Languages (Gala et al., 2023)](https://arxiv.org/abs/2305.16307) — Architecture and evaluation of IndicTrans2; the recommended translation tool for Tamil NL→SQL data construction.

[Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model (Üstün et al., 2024)](https://arxiv.org/abs/2402.07827) — 101-language instruction-tuned model from Cohere; useful reference for what a properly multilingual fine-tuning pipeline looks like.

[RAT-SQL: Relation-Aware Schema Encoding and Linking for Text-to-SQL Parsers (Wang et al.)](https://arxiv.org/abs/1911.04942) — Direct study of cross-lingual NL→SQL transfer; covers what enables and limits transfer across language pairs.

[CodeGeeX: A Pre-Trained Model for Code Generation with Multilingual Benchmarking (Zheng et al. 2023)](https://arxiv.org/abs/2303.17568) — Analysis of code model performance on multilingual instruction following; relevant for understanding Qwen2.5-Coder's Tamil limitations.

## Videos

- [AI4Bharat — IndicTrans2 Overview](https://www.youtube.com/watch?v=_oNDAfvJR6o) — AI4Bharat — ~30 min. Official walkthrough of IndicTrans2 usage for Indian language translation; covers Tamil specifically.
- [Yannic Kilcher — Multilingual Models and Cross-Lingual Transfer](https://www.youtube.com/watch?v=dMD8QPCp38Y) — Yannic Kilcher — ~40 min. Conceptual overview of how multilingual representations enable cross-lingual transfer.

## Blog Posts / Articles

[AI4Bharat — Samanantar: The Largest Publicly Available Parallel Corpus for Indic Languages](https://huggingface.co/datasets/ai4bharat/samanantar) — Dataset page; includes Tamil-English parallel sentences that can be filtered for question-type text to supplement your training data.

[Hugging Face — Multilingual Tokenization Guide](https://huggingface.co/docs/tokenizers/components#models) — Covers BPE, WordPiece, and SentencePiece tokenization and how vocabulary composition affects non-English languages.

[Sebastian Ruder — A Survey of Cross-Lingual Embedding Methods](https://ruder.io/cross-lingual-embeddings/) — Foundational reading on how cross-lingual representations work; directly explains why transfer from English SQL training to Tamil inference is possible at all.

## GitHub Repos

[AI4Bharat/IndicTrans2](https://github.com/AI4Bharat/IndicTrans2) — Official repo with installation instructions, model downloads, and inference code for Tamil-English translation.

[google-research/xtreme](https://github.com/google-research/xtreme) — Cross-lingual transfer benchmark; useful for understanding where Tamil stands relative to other languages in terms of model coverage.

[Helsinki-NLP/opus-mt-en-ta](https://huggingface.co/Helsinki-NLP/opus-mt-en-ta) — Lightweight English→Tamil model on HuggingFace; faster alternative to IndicTrans2 for bulk translation, lower quality for technical text.

## Documentation

[HuggingFace — Tokenizer Vocabulary Extension](https://huggingface.co/docs/transformers/internal/tokenization_utils#transformers.PreTrainedTokenizer.add_tokens) — How to add new tokens with `add_tokens()` and resize the model embedding matrix.

[Unicode Tamil Script Block (U+0B80–U+0BFF)](https://www.unicode.org/charts/PDF/U0B80.pdf) — Official Unicode chart for Tamil characters; useful when debugging tokenization of individual glyphs.

## Optional / Bonus

[OPUS — Open Parallel Corpus (Tamil-English)](https://opus.nlpl.eu/) — Large collection of parallel Tamil-English text across domains; filter for question-type sentences to build additional NL→SQL training data.

[Dravidian-CodeMix NLP at FIRE 2021](https://dravidian-codemix.github.io/2021/index.html) — Tamil code-mixing shared task; relevant if your users mix Tamil and English words in a single question (a common pattern in conversational interfaces).
