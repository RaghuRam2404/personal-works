# Week 39 Resources — Execution-Based Evaluation

## Papers

[Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task](https://arxiv.org/abs/1809.08887) — Yu et al. 2018. The benchmark that defined execution-based eval for text-to-SQL; read Section 4 on evaluation methodology.

[BIRD: A Big Bench for Large-Scale Database Grounded Text-to-SQLs](https://arxiv.org/abs/2305.03111) — Li et al. 2023. Extends Spider with dirtier real-world schemas; introduces execution accuracy as the primary metric.

[Evaluating the Text-to-SQL Capabilities of Large Language Models](https://arxiv.org/abs/2204.00498) — Rajkumar et al. 2022. Systematic analysis of LLM text-to-SQL with execution-based metrics; useful for understanding baseline expectations.

## Videos

[Text-to-SQL Evaluation Deep Dive — Defog.ai](https://www.youtube.com/watch?v=UcIc4YMuWOo) — Defog AI channel. ~25 min. Walks through the sql-eval framework, execution correctness implementation, and real benchmarking results on open-source models.

[Running PostgreSQL in Google Colab](https://www.youtube.com/watch?v=R_hLXB5GnmI) — Data Science Simplified. ~12 min. Step-by-step Colab setup for Postgres with psycopg2.

## Blog Posts / Articles

[How We Evaluate Text-to-SQL Models at Defog](https://defog.ai/blog/how-we-evaluate-text-to-sql-models/) — Defog AI blog. Explains the design decisions behind their evaluation harness, including result set normalization and safety filtering.

[Why Exact Match is a Bad Metric for Text-to-SQL](https://towardsdatascience.com/why-exact-match-is-a-bad-metric-for-text-to-sql-evaluation-5c9c8e1e4c3a) — Towards Data Science. Makes the quantitative case for execution-based eval with concrete examples.

## GitHub Repos

[defog-ai/sql-eval](https://github.com/defog-ai/sql-eval) — Defog's open-source execution-based evaluation framework for text-to-SQL. Contains schema creation, data generation, SQL execution, and result comparison. Start here before writing your own harness.

[taoyds/spider](https://github.com/taoyds/spider) — Official Spider benchmark repo. Includes the evaluation scripts that defined the field-standard execution accuracy metric.

[psycopg/psycopg2](https://github.com/psycopg/psycopg2) — The Python PostgreSQL adapter you use in your harness. The `cursor.execute` and `fetchall` patterns used this week are its core API.

## Documentation

[PostgreSQL: statement_timeout](https://www.postgresql.org/docs/current/runtime-config-client.html#GUC-STATEMENT-TIMEOUT) — Official Postgres docs on query timeout configuration. Shows how to set per-session and per-transaction timeouts.

[psycopg2: Basic module usage](https://www.psycopg.org/docs/usage.html) — Covers connect, cursor, execute, fetchall, and error handling. The patterns for your harness are all in the first two sections.

[sqlparse: Parsing SQL](https://sqlparse.readthedocs.io/en/latest/analyzing.html) — Docs for the `sqlparse` library used in `is_safe_sql`. Shows how `get_type()` works and its edge cases with CTEs.

## Optional / Bonus

[DAIL-SQL: Efficient Prompt Engineering for Text-to-SQL](https://arxiv.org/abs/2308.15363) — Gao et al. 2023. State-of-the-art prompting approach for text-to-SQL; discusses how their eval harness handles edge cases. Relevant if you want to compare your fine-tuned model against a strong prompting baseline.

[The SQL Sandbox Pattern](https://brandur.org/fragments/sql-sandbox) — Brandur Leach. Short blog post on the schema isolation pattern used in test harnesses. Confirms the design choice you made this week.
