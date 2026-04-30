# Week 25 Glossary — Dataset Construction

**Alpaca format**: A single-turn instruction dataset format with `instruction`, `input`, and `output` fields; used by Stanford Alpaca and many early fine-tuned models.

**ShareGPT format**: A multi-turn conversation dataset format with a `conversations` list of `{"from": "human/gpt", "value": ...}` dicts; used by Vicuna and OpenHermes datasets.

**ChatML format**: The current standard conversation format using `<|im_start|>role\ncontent<|im_end|>` tokens; used by OpenAI, Qwen2.5, and most modern instruction-tuned models.

**Self-Instruct**: A technique (Wang et al. 2022) that uses a language model to generate new training instructions from a seed set of hand-written examples, enabling large-scale synthetic dataset creation.

**System prompt**: The first message in a ChatML conversation that defines the model's persona, capabilities, and output format constraints.

**Loss masking**: During fine-tuning, computing the cross-entropy loss only on assistant tokens (not user or system tokens) to train the model to generate responses, not questions.

**DataCollatorForCompletionOnlyLM**: HuggingFace TRL's data collator that automatically masks the loss for non-assistant tokens in ChatML conversations.

**sqlglot**: A Python library for SQL parsing and transpilation across dialects (PostgreSQL, MySQL, SQLite, BigQuery, etc.); used to validate and convert SQL.

**Spider benchmark**: A cross-domain text-to-SQL dataset with 200+ databases and 10,000+ NL→SQL pairs; uses SQLite-compatible SQL.

**BIRD benchmark**: A harder text-to-SQL benchmark with real enterprise databases; more complex queries than Spider.

**WikiSQL**: An early (2017) text-to-SQL dataset with 80K examples but limited to single-table, grammar-restricted queries; largely superseded by Spider and BIRD.

**Schema conditioning**: Including CREATE TABLE statements or column descriptions in the input prompt so the model can generate SQL referencing the correct column and table names.

**Tier system**: A dataset construction strategy where examples are categorized by quality tier (human-written > human-curated > synthetic) and potentially given different training weights.

**Template filling**: A data augmentation technique where a hand-written example template is varied (different table names, values, conditions) to create multiple controlled-quality examples.

**sqlglot dialect**: A language specification in sqlglot for parsing SQL from a specific database (`"postgres"`, `"mysql"`, `"sqlite"`, etc.).
