# Week 61 Glossary

**Spider:** A cross-domain text-to-SQL benchmark (Yale, 2018) with 10,206 question/SQL pairs across 206 databases; the standard baseline benchmark for text-to-SQL research.

**BIRD-SQL:** A text-to-SQL benchmark (2023) with 12,751 harder questions including external knowledge requirements, complex SQL, and realistic database content.

**Spider 2.0:** A 2024 evolution of Spider targeting enterprise-scale databases (BigQuery, Snowflake, PostgreSQL) with multi-step reasoning and agent-loop evaluation.

**Defog sql-eval:** An open-source evaluation framework and dataset from Defog.ai measuring real-world enterprise SQL generation quality.

**Execution accuracy:** The fraction of model-generated SQL queries that produce the same result set as the reference SQL when run against the test database; the primary text-to-SQL evaluation metric.

**Exact match accuracy:** The fraction of model-generated SQL queries that are token-for-token identical to the reference SQL; a stricter, more brittle metric than execution accuracy.

**Dev set:** The held-out evaluation split used during model development and selection; contrasted with the test set, which should only be evaluated once for final reporting.

**Test set contamination:** Using test set performance to guide model selection or hyperparameter tuning; even without training on test examples, this inflates reported scores.

**Bootstrapping:** A statistical resampling technique for estimating confidence intervals by sampling the evaluation set with replacement 1,000+ times and computing the statistic on each sample.

**Confidence interval (CI):** A range of values [lo, hi] such that the true population metric falls within the range with a specified probability (e.g., 95%); essential for comparing models whose scores are close.

**External knowledge (BIRD):** Domain-specific information provided in the question context that must be extracted and used in SQL generation (e.g., "burnout risk score > 0.7" as a threshold).

**Cross-database generalization:** The ability to generate correct SQL for database schemas not seen during training; the key capability measured by Spider and BIRD.
