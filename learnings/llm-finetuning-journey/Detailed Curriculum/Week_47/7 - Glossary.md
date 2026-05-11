# Week 47 Glossary

**Reward hierarchy**: A multi-level reward scheme that assigns graded scores (e.g., 0.0, 0.1, 0.2, 0.5, 1.0) based on increasing levels of SQL correctness; reduces sparsity compared to binary rewards.

**Reward hacking (SQL-specific)**: When the model discovers SQL patterns that score high on the reward function without being good SQL — e.g., querying information_schema, using SELECT * without WHERE, or generating very long outputs to trigger a reasoning bonus.

**Anti-hack guard**: An explicit check in the reward function that returns a fixed low score for known reward-hacking patterns, regardless of execution success.

**Extract_sql**: A function that parses model output text to extract only the SQL query, handling code fences (```sql), markdown formatting, and prose explanations before the SQL.

**Reward contract**: The set of requirements a reward function must satisfy: deterministic, fast (< 500ms/query), safe (read-only), and correct (reward 1.0 means genuinely correct).

**Diagnostic test**: Running the reward function on a sample of model completions before GRPO training begins; verifies the reward distribution is non-degenerate.

**Zero-gradient prompt**: A GRPO training prompt where all K completions receive the same reward; produces no gradient update; common for trivially easy or hard prompts.

**Shaped reward**: A reward function with multiple distinct levels that provide gradient signal even for partially-correct outputs; contrasts with binary {0,1} reward.

**Semantic correctness**: The SQL query returns the same rows as the reference query; stronger than execution correctness (which only checks that the query runs without error).

**Structural correctness**: The SQL query has the correct schema (column names, number of rows) even if some values differ; intermediate between execution correctness and semantic correctness.

**Row set equality**: Comparing `set(actual_rows) == set(expected_rows)` for exact-match scoring; misses duplicate rows and requires conversion to comparable types (tuples).

**Reasoning bonus**: A small additional reward for generating a reasoning chain before the SQL; must be kept small (≤ 10% of max reward) to avoid verbosity hacking.
