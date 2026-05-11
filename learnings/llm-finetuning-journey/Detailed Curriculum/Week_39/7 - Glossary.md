# Week 39 Glossary

**Execution-based evaluation**: Evaluating SQL correctness by executing the query and comparing result sets, not token strings.

**Exact match**: A binary metric that is 1 only when the generated SQL is character-for-character identical to the reference SQL.

**Execution success rate**: Percentage of model-generated SQL queries that execute without a runtime error.

**Execution correctness**: Percentage of model-generated SQL queries that execute AND return result sets matching the expected output.

**Eval harness**: A pipeline that loads test examples, runs inference, executes SQL, compares outputs, and records metrics.

**Schema isolation**: Creating a fresh database schema per test example to prevent state leakage between tests.

**Statement timeout**: A Postgres setting that aborts a query exceeding a wall-clock time limit (`SET statement_timeout = 5000` for 5 seconds).

**Result set normalization**: Converting query output rows to a canonical form (stringify, sort, NULL-replace) before comparison.

**Non-SELECT rejection**: A safety filter that refuses to execute any SQL that is not a SELECT statement.

**Semantic error**: Generated SQL that executes successfully but returns wrong rows (correct syntax, wrong meaning).

**Structural error**: Generated SQL that fails to execute due to referencing non-existent tables or columns.

**Error taxonomy**: A categorization of model failures (wrong filter, wrong join, wrong aggregation, syntax error) used to prioritize training data improvements.

**Deterministic test data**: Synthetic rows inserted with a fixed random seed so the expected result set is reproducible across runs.
