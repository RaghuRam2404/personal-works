# Week 40 Glossary

**Phase gate**: A checkpoint that requires specific deliverables to be complete before progressing to the next phase.

**Model card**: A structured document on HuggingFace Hub describing a model's training, evaluation, limitations, and usage.

**HuggingFace Hub**: The model and dataset registry where you publish and version ML artifacts for reuse.

**Adapter push**: Uploading a LoRA adapter (not the full base model) to the Hub — typically a few hundred MB rather than 14 GB.

**Dataset card**: A documentation artifact on HuggingFace Datasets describing data provenance, schema, splits, and intended use.

**Held-out test set**: A benchmark set never used in training or validation; kept private to provide an unbiased final evaluation.

**Model retrospective**: A written reflection on what was learned, what surprised you, and what gaps remain — a standard engineering practice for long projects.

**Inference pipeline**: The complete sequence of steps from raw user input to final output, including prompt construction, tokenization, generation, and post-processing.

**Data leakage**: Accidental inclusion of test examples in training data, which inflates evaluation metrics and invalidates the benchmark.

**Phase 5 preview**: GRPO (Group Relative Policy Optimization) — reward-based fine-tuning using the execution harness as the verifier; the primary technique for pushing exec correctness beyond the SFT plateau.
