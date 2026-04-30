# Week 28 Glossary

**Continued pretraining**: Resuming next-token prediction training on domain text using an already-pretrained base model, without any labeled pairs.

**Supervised fine-tuning (SFT)**: Training a pretrained model on labeled (input, output) pairs using cross-entropy loss computed only on output tokens.

**Instruction tuning**: A specific form of SFT where data is formatted as (instruction, response) pairs; teaches general instruction-following behavior.

**Post-training pipeline**: The sequence of training stages applied after pretraining: typically SFT → preference optimization (DPO) → reward optimization (GRPO).

**Input masking**: Setting loss labels to -100 for prompt tokens during SFT so the model is only penalized for errors on response tokens.

**Reward model**: A model trained on human preference rankings to assign a scalar score to (prompt, response) pairs; used in RLHF between SFT and PPO stages.

**Catastrophic forgetting**: Degradation of pretrained general capabilities when fine-tuning overwrites weight subspaces that encode broad language understanding.

**Intrinsic low-rank hypothesis**: The empirical observation that weight change matrices during fine-tuning have low effective rank, justifying LoRA's approximation.

**DPO (Direct Preference Optimization)**: A training method that optimizes model outputs toward human preferences without an explicit reward model, using (prompt, chosen, rejected) triples.

**GRPO (Group Relative Policy Optimization)**: A reinforcement learning method using verifiable scalar rewards (e.g., SQL execution correctness) to optimize model outputs.

**Base model**: A model trained only on next-token prediction on large unlabeled corpora, before any fine-tuning; a "document completer," not an assistant.

**Teacher forcing**: During SFT training, providing the ground-truth previous tokens as input at each step rather than the model's own previous prediction; standard in cross-entropy training.
