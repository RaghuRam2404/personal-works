# Week 26 Glossary — Domain Dataset Construction

**postgres-sql-v1**: The name of your first-version PostgreSQL/TimescaleDB text-to-SQL fine-tuning dataset; 5,000 examples in ChatML format.

**Execution accuracy (EA)**: The primary evaluation metric for text-to-SQL models; the percentage of generated SQL queries that execute and return the correct result when run against the database.

**Schema consistency**: The property that every table and column referenced in generated SQL exists in the schema provided in the prompt; a necessary condition for executable SQL.

**Cross-deduplication**: Removing near-duplicate examples across tiers (e.g., removing Tier 3 examples that are near-identical to Tier 1 examples); ensures training examples are genuinely distinct.

**time_bucket_gapfill()**: A TimescaleDB function that fills in missing time intervals with NULL values; extends `time_bucket()` to include time periods with no data.

**psycopg2**: The standard Python adapter for PostgreSQL; used for executing SQL verification queries against a real database.

**DataCollatorForCompletionOnlyLM**: HuggingFace TRL's data collator that masks loss on non-assistant tokens in ChatML conversations; essential for correct fine-tuning loss computation.

**Dataset card (README.md)**: The documentation file for a HuggingFace Hub dataset; describes dataset purpose, construction methodology, statistics, and intended use.

**push_to_hub()**: HuggingFace datasets method that uploads a local dataset to the HuggingFace Hub repository.

**80/20 split**: Dividing a dataset into 80% training and 20% validation; the standard split for fine-tuning datasets.

**Over-represented pattern**: A SQL construct that appears disproportionately often in the training data, causing the model to over-generate it relative to other equally valid constructs.

**Semantic alias error**: A fine-tuning failure mode where the model learns the wrong interpretation of a natural language term (e.g., "total value" → single column sum instead of quantity × price).

**DatasetDict**: A HuggingFace datasets class that holds multiple dataset splits (train, validation, test) as a dictionary.

**Paraphrase pairs**: Training examples where the same underlying SQL query is paired with multiple different natural language question phrasings; improves model robustness to question variation.
