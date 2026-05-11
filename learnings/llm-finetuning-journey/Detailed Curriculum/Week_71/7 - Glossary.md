# Week 71 Glossary — Frontier Reading: Tulu 3, SmolLM2, OLMo 2

**RLVR (Reinforcement Learning with Verifiable Rewards)**: A training paradigm using deterministic, objective reward functions (code execution, math verification) instead of learned reward models; introduced as the core of Tulu 3's alignment stage.

**On-policy data generation**: Using the current training checkpoint to generate new training examples, then verifying and re-adding them to the dataset; creates a self-improvement loop when combined with external verification.

**Model collapse**: The degradation of a model's output diversity that occurs when training on model-generated data without an external quality filter; the model amplifies its own errors and narrows its output distribution.

**FineWeb-Edu**: A high-quality web text dataset filtered for educational value using a classifier; used as SmolLM2's primary pretraining data source.

**DCLM (DataComp for Language Models)**: A benchmark and dataset suite for evaluating data curation strategies for pretraining; used in SmolLM2 training.

**Mid-training phase**: A second pretraining stage that upweights high-quality domain data (code, StackExchange, Wikipedia) after broad pretraining; introduced by OLMo 2 as a transition step between pretraining and fine-tuning.

**OLMES**: OLMo's open evaluation harness; a fully reproducible evaluation framework released with OLMo 2.

**Tokenizer efficiency**: The degree to which a tokenizer represents domain-specific vocabulary (e.g., SQL keywords) as single tokens rather than multi-token sequences; affects effective sequence length and inference speed.

**Capability emergence**: The non-linear development of specific skills during training, where a capability appears sharply at a particular training step rather than gradually; only observable with intermediate checkpoints.

**Dolmino Mix**: OLMo 2's pretraining data mixture; fully released for reproducibility; includes curated web text, code, and Wikipedia.
