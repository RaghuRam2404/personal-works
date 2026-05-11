# Week 37 Glossary

**Domain-adaptive fine-tuning**: Continuing fine-tuning with task-specific or domain-specific data to improve performance on a narrow target distribution; distinct from continued pretraining (no labels) and general SFT.

**Synthetic data generation**: Using a large language model (Claude, GPT-4) to generate training examples from prompts; enables creating examples for rare features or specialized domains without manual annotation.

**sqlparse**: A Python library for parsing and validating SQL statements; used to filter out invalid SQL examples from training datasets.

**Data contamination**: The presence of test set examples (or near-duplicates) in the training set; inflates evaluation metrics and gives a misleadingly optimistic view of generalization.

**Deduplication**: Removing exact or near-duplicate examples from a dataset to prevent the model from memorizing repeated patterns and to ensure training efficiency.

**MySQL-specific syntax**: SQL syntax features exclusive to MySQL (e.g., backtick identifiers, `AUTO_INCREMENT`, `TINYINT(1)`) that are invalid in PostgreSQL; must be filtered from PostgreSQL training datasets.

**Schema diversity**: Having a wide variety of table structures, column types, relationships, and naming conventions in the training dataset; essential for generalization to novel schemas at inference time.

**TimescaleDB**: An open-source time-series database built on PostgreSQL; adds features like `time_bucket`, hypertables, and continuous aggregates. Your primary target domain for advanced SQL generation.

**`time_bucket`**: A TimescaleDB function that buckets timestamps into intervals (e.g., `time_bucket('1 hour', time)` groups records by hour); a key feature for time-series SQL generation.

**SQL type distribution**: The fraction of training examples representing each SQL operation type (simple SELECT, JOIN, GROUP BY, subquery, etc.); should be balanced to avoid over-specialization.

**Data curation pipeline**: The sequence of steps applied to raw dataset sources to produce clean, validated, deduplicated, formatted training data; the engineering foundation of domain fine-tuning.
