# Week 8 — Glossary

Terms introduced specifically in the context of Week 8 (Phase Gate and capstone concepts). For all prior terms, see Weeks 1–7 Glossaries.

**Phase Gate**: A required checkpoint at the end of each phase; a set of criteria you must pass before advancing to the next phase. Failing a gate means repeating weak modules.

**Capstone project**: An end-to-end project synthesizing all skills from a phase; serves as both a learning integration exercise and a portfolio artifact.

**nanoGPT**: Karpathy's minimal GPT implementation (~300 lines); a reference architecture for a decoder-only transformer. You will study every line in Phase 2.

**char-level transformer**: A transformer language model operating on individual characters rather than subword tokens; longer sequences, smaller vocabulary.

**GPTConfig**: Configuration dataclass for nanoGPT specifying `n_layer`, `n_head`, `n_embd`, `block_size`, `vocab_size`, `dropout`; analogous to HuggingFace's `PretrainedConfig`.

**block_size**: In nanoGPT, the maximum sequence length (context window); equivalent to `max_position_embeddings` in HuggingFace models.

**timed test**: Self-assessment exercise of writing a complete training loop from memory within a fixed time limit (10 minutes for Phase 1); a measure of internalization.

**schema-grounded generation**: SQL generation conditioned on the database schema (table/column names and types); the gap between the capstone model and a real text-to-SQL system.

**teacher forcing (review)**: During training, feeding ground-truth tokens as context rather than model predictions; introduced Week 4 but critical to understanding why your capstone model generates gibberish column names.

**temperature (review)**: Logit scaling parameter during generation (introduced Week 4); 0.8 is the recommended starting value for SQL generation in the capstone.
