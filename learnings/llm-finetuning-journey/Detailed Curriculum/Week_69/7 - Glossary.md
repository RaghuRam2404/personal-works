# Week 69 Glossary — Evaluation and Ablations

**Exact-match (EM) accuracy**: The fraction of generated SQL strings that match the reference SQL string exactly after normalization; strict but easy to compute without a live database.

**Execution accuracy (EX)**: The fraction of generated SQL queries that produce the same result set as the reference SQL when run against the test database; more meaningful than EM but requires a running database.

**Normalization (SQL evaluation)**: A sequence of transformations applied to SQL strings before comparison: lowercasing, whitespace collapsing, semicolon removal, alias stripping.

**Baseline**: A model or system used as a comparison point in evaluation; must be identified with exact version strings and evaluation dates.

**Confidence interval (CI)**: A range around a reported accuracy that quantifies estimation uncertainty due to finite test set size; for a 200-example binary test at 83%, the 95% CI is approximately ±5 pp.

**Ablation study**: A controlled experiment where one component of a pipeline is removed to measure its isolated contribution; each row in the ablation table represents a different configuration.

**Additive ablation**: An ablation table structured so each row adds exactly one component to the previous row; makes individual contributions interpretable.

**Failure mode**: A specific category of error that a model makes; identifying and quantifying failure modes guides future data collection and training improvements.

**Schema hallucination**: A model error where the generated SQL references a column or table that does not exist in the provided schema.

**† (dagger) notation**: Convention in ML results tables to mark results taken from published sources rather than measured by the paper's authors.
