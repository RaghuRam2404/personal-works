# Week 51 Glossary

**Diminishing returns**: When each successive iteration experiment produces smaller improvements than the last; the signal to stop iterating and move to Phase 6.

**Model selection framework**: A structured set of criteria (execution accuracy, semantic accuracy, complex query performance, KL divergence, generation length) for choosing among multiple model checkpoints.

**Blind validation set**: A set of evaluation queries that was never used to select hyperparameters or guide experiments; provides an unbiased estimate of generalization performance.

**Eval set overfitting**: The implicit bias introduced by repeatedly running experiments that improve metrics on the same evaluation set; the selected model may be specifically better on those queries, not in general.

**Phase 5 best model**: The checkpoint selected at the end of Week 51 that will be submitted to the Phase 5 Gate and used as the starting point for Phase 6 training.

**Residual gap**: SQL query types or complexity levels where the best Phase 5 model still underperforms; these gaps define the Phase 6 dataset construction priorities.

**Model card**: A documentation page (usually on HuggingFace Hub) describing a model's training procedure, intended use, evaluation results, and known limitations.

**Iteration ceiling**: The maximum performance achievable with the current training data distribution; reached when more GRPO steps or hyperparameter changes produce no further improvement.

**Version tag**: A HuggingFace Hub model identifier suffix (e.g., `-phase5-best`) that identifies the model's role in the training pipeline and makes it easily retrievable.

**Gate criterion**: A specific, measurable requirement that must be met to advance from one Phase to the next; for Phase 5, this is v3 beating v1 on execution accuracy.
