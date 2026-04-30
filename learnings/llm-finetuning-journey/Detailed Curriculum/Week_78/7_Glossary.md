# Week 78 Glossary — Final Phase 6 Gate and Course Wrap

**Capstone audit:** A structured checklist review of all promised deliverables to verify completion before declaring a project done.

**Model card:** A standardized document accompanying a published model that specifies its purpose, training procedure, evaluation results, usage instructions, and limitations.

**Reproducibility:** The property that a result can be independently verified by another researcher following the published method with the published artifacts.

**Benchmark contamination:** When the benchmark used for evaluation overlaps with data seen during training, inflating apparent performance.

**Ablation study:** A systematic experiment that isolates the contribution of each component or stage by removing one at a time and measuring impact.

**arXiv:** An open-access preprint server (arxiv.org) where ML researchers publish work before or instead of formal peer review; cs.CL is the relevant track for NLP.

**Endorsement (arXiv):** A requirement for new submitters to arXiv to have an existing arXiv author vouch for their submission; required in most ML subfields.

**Portfolio artifact:** Any externally accessible, linkable evidence of technical work (model on HuggingFace, code on GitHub, paper on arXiv, blog post); distinct from work that exists only locally.

**Task specificity trade-off:** The inverse relationship between a model's performance on a narrow domain and its performance on adjacent tasks not covered by training data.

**Execution accuracy (EX):** An evaluation metric for NL→SQL that checks whether the executed result set of predicted SQL matches the executed result set of gold SQL, rather than requiring exact string match.

**Speculative decoding:** An inference acceleration technique where a smaller draft model generates tokens in parallel and a larger model verifies them in batch, reducing latency.

**Query cache:** An inference optimization that stores (question → SQL) mappings and serves cached results for semantically similar repeated questions, bypassing model generation.

**Domain benchmark:** An evaluation dataset designed to measure performance on a specific domain or dialect; Custom-200 is your TimescaleDB domain benchmark.
