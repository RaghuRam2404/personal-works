# Week 55 Glossary

**LLM-as-judge:** Using a strong language model (e.g., GPT-4) to evaluate the quality of model outputs or training data, replacing or augmenting human annotation.

**Alpagasus:** A 2023 paper demonstrating that filtering Stanford Alpaca's 52K dataset to 9K high-scoring examples (using GPT-4 as judge) produces a stronger fine-tuned model.

**Cohen's kappa:** A statistic measuring inter-rater agreement that corrects for chance agreement; values > 0.6 indicate substantial agreement.

**Calibration set:** A small, human-annotated sample used to verify that an automated judge's decisions align with human judgment before scaling to the full dataset.

**Rubric:** An explicit scoring guide with definitions for each score level; essential for consistent LLM judge behavior.

**Inter-rater agreement:** The degree of consensus between two independent raters (human vs. LLM, or LLM vs. LLM) when labeling the same examples.

**Verbosity bias:** The tendency of LLM judges to rate longer, more elaborate outputs higher regardless of correctness; a known systematic error in LLM judging.

**Multi-signal filter:** A filtering pipeline that combines multiple independent quality signals (execution, deduplication, LLM score) rather than relying on any single criterion.

**Skill-adaptive threshold:** Applying a more lenient quality cutoff for rare skills to prevent them from being eliminated entirely by aggressive filtering.

**Attrition rate:** The fraction of examples removed at each filtering stage; used to audit the pipeline and diagnose over-aggressive filters.

**Semantic consistency check:** A filtering step that compares a generated SQL query's result set against a reference answer's result set to detect semantically wrong but syntactically valid queries.

**Data-centric AI:** The philosophy that improving data quality and curation is more impactful than tuning model architecture or training hyperparameters.
