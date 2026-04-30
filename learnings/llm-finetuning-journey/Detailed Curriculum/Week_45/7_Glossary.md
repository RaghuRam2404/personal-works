# Week 45 Glossary

**Execution accuracy**: The fraction of generated SQL queries that execute without error on the database; a necessary but not sufficient condition for correctness.

**Semantic accuracy**: The fraction of generated SQL queries that return the same rows as the reference SQL; measures actual correctness, not just syntactic validity.

**Syntax error rate**: The fraction of generated SQL queries that fail with a syntax error (parsing failure before execution); the simplest DPO improvement target.

**Empty result rate**: Generated SQL executes successfully but returns zero rows; often indicates a wrong WHERE clause or JOIN condition.

**Length normalization (DPO)**: A DPO variant that divides the log-probability sum by sequence length before computing the loss; prevents the model from favoring short completions whose low per-token probability is hidden by length.

**Reward margin plateau**: When reward_margin stops growing during DPO training; caused by mislabeled data, high β, or learning rate too high.

**LoRA r (rank)**: The rank of the low-rank decomposition matrices; higher rank = more trainable parameters = more expressivity. Typical: r=16 for DPO.

**Unsloth DPO**: A memory-optimized implementation of TRL's DPOTrainer that avoids the dual-model memory spike; ~2–3× faster than vanilla TRL for 7B models.

**Semantic mislabeling**: When a preference pair's "chosen" SQL executes without error but returns wrong rows; a common issue when execution-only labeling is used without reference output comparison.

**Eval report**: A Markdown document comparing model versions across metrics (execution accuracy, semantic accuracy, syntax error rate, etc.) broken down by query complexity tier.
