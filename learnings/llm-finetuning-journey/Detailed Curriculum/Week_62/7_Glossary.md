# Week 62 Glossary

**McNemar's test:** A statistical test for comparing two classifiers on paired examples; tests whether the number of examples where model A is correct and B is wrong significantly exceeds the reverse; appropriate for comparing model accuracy on the same test set.

**Discordant pairs:** In McNemar's test, pairs of examples where the two models disagree (one correct, one wrong); the basis of the test statistic.

**API caching:** Storing API call responses keyed by input hash; prevents redundant API calls and enables reproducible evaluation without additional cost.

**Cost-per-correct-query:** A composite efficiency metric: inference cost per query divided by accuracy; more useful than accuracy alone for production decision-making.

**Domain advantage hypothesis:** The claim that a domain-specialized model will outperform generalist models on in-domain tasks even if it underperforms on general benchmarks.

**Error taxonomy:** A classification of model failures by root cause (schema hallucination, wrong logic, wrong function, timeout, etc.); essential for directing future improvements.

**Type A error (schema hallucination):** Model generates SQL referencing columns or tables not in the given schema.

**Type B error (wrong logic):** Model generates valid SQL with the correct schema references but wrong filtering, aggregation, or join logic.

**Type C error (wrong function):** Model generates SQL with correct intent but wrong function syntax (e.g., wrong TimescaleDB hyperfunction arguments).

**Hybrid deployment:** A production architecture that routes queries through a cheap local model first and falls back to an expensive API model only when the local model fails execution; achieves high accuracy at low cost.

**Break-even analysis:** Computing the cost threshold at which two deployment options have equal total cost; used to justify the economic case for a locally-deployed specialized model.

**Statistical significance:** A formal threshold (typically p < 0.05) for concluding that an observed performance difference is unlikely to be due to chance; always required when comparing model performance.
