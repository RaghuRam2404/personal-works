# Week 60 Glossary

**GRPO (Group Relative Policy Optimization):** A reinforcement learning method for LLMs that generates K candidate responses per prompt, computes rewards for each, and uses within-group reward differences as advantage estimates — eliminating the need for a separate value/critic network.

**Group size (K):** The number of candidate completions generated per prompt in GRPO; higher K gives better advantage estimates but costs K× the inference compute.

**Advantage estimate:** The difference between a candidate's reward and the group mean reward; used as the RL signal in GRPO. Zero advantage → zero gradient → no learning.

**KL coefficient (kl_coef):** The weight on the KL-divergence penalty term in GRPO; controls how far the policy is allowed to deviate from the reference model.

**Reward variance:** The spread of rewards within a GRPO group; zero variance (all rewards equal) produces no gradient; high variance produces strong learning signal.

**Partial credit reward:** A reward function that assigns fractional rewards for near-correct outputs (e.g., SQL that executes but returns wrong rows); produces denser gradients than binary 0/1 rewards.

**Pass@K:** A metric measuring the fraction of prompts for which at least 1 of K sampled responses is correct; represents the ceiling of what GRPO can achieve with group size K.

**Merged model:** A model checkpoint where LoRA adapter weights have been mathematically combined with the base model weights into a single weight matrix; produces a standalone model file without the separate LoRA files.

**save_method="merged_16bit":** Unsloth's function for merging LoRA adapters into the base model at bfloat16 precision; produces a HuggingFace-compatible model ready for inference without adapter loading.

**Distribution collapse:** A failure mode in RL fine-tuning where the policy becomes overconfident and generates only a narrow set of outputs; monitored by tracking KL divergence from the reference model.

**Executable reward:** A reward signal derived from actually running the model's output (e.g., executing generated SQL against a database); the most reliable reward for code generation tasks because it is objective and non-gameable.

**Reference model (GRPO):** The frozen policy used in GRPO's KL constraint; typically the DPO model that precedes GRPO training.
