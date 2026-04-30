# Week 38 Glossary

**Production training run**: A full training run with carefully chosen hyperparameters, validation monitoring, checkpointing, and documentation; as opposed to exploratory runs for tuning.

**`max_grad_norm`**: Gradient clipping parameter; gradients whose norm exceeds this value are rescaled to have exactly this norm. Prevents loss spikes from anomalous batches. Standard value: 1.0.

**`load_best_model_at_end`**: `SFTConfig` option that reloads the checkpoint with the best evaluation metric at the end of training; requires `evaluation_strategy` to be set. Implements implicit early stopping.

**Model card**: A documentation file (README on HuggingFace Hub) describing a model's intended use, training data, methodology, performance, and limitations; required for responsible model sharing.

**Exact match accuracy**: The fraction of model-generated SQL outputs that exactly match the reference SQL string; a conservative lower bound on true quality (semantically equivalent queries may differ textually).

**Execution correctness**: A more permissive evaluation metric that runs generated SQL against a database and checks whether the result set matches the expected result; allows for syntactic variations in semantically equivalent queries.

**Error analysis**: Systematic categorization of model mistakes by error type; used to identify the most impactful dataset or training improvements for the next iteration.

**`postgres-sqlcoder-7b-v1`**: Your first production-quality PostgreSQL text-to-SQL model; the Phase 4 culmination artifact. Will be the starting point for Phase 5's preference optimization.

**HuggingFace Hub**: The central repository for sharing machine learning models, datasets, and spaces; hosts both the base models you fine-tune and the adapters you publish.

**Adapter-only push**: Pushing only the LoRA adapter weights (50–100MB) rather than the full merged model (14GB); allows iterative updates without duplicating base model storage.
