# Week 46 Glossary

**GRPO (Group Relative Policy Optimization)**: Policy gradient algorithm that replaces the critic network with within-group reward normalization; introduced in DeepSeekMath.

**RLVR (Reinforcement Learning with Verifiable Rewards)**: Design pattern for RL training where the reward function is a deterministic verifier (test runner, SQL executor, math checker) rather than a learned reward model.

**Group (GRPO)**: The K completions generated for a single prompt; within-group statistics (mean, std of rewards) provide the advantage baseline.

**Group-relative advantage**: (r_i − mean(r)) / std(r) for completion i in a group; the normalized reward signal used in GRPO.

**num_generations (K)**: The number of completions sampled per prompt in GRPO; typical values are 4, 8, or 16.

**Verifier**: A deterministic function that evaluates correctness of a model output; for SQL, this is the database executor; for math, it checks if the answer equals the expected value.

**Verifiable reward**: A reward signal produced by a verifier rather than a learned reward model; cannot be reward-hacked in the traditional sense because the verifier is ground truth.

**DeepSeekMath**: 2024 paper by Shao et al. introducing GRPO as an algorithm for mathematical reasoning.

**DeepSeek-R1**: 2025 paper demonstrating that GRPO + verifiable rewards on math problems produces a model matching OpenAI o1 on reasoning benchmarks, while revealing emergent chain-of-thought reasoning.

**Reasoning chain emergence**: The spontaneous development of extended step-by-step reasoning in a model trained only on final-answer verifiable rewards; observed in DeepSeek-R1.

**Cold start (DeepSeek-R1)**: An initialization strategy that fine-tunes on a small set of high-quality reasoning examples before GRPO training, to avoid the model exploring incoherent reasoning paths early in training.

**RLHF (vs. RLVR)**: RLHF uses human preference labels as rewards (subjective, expensive); RLVR uses deterministic verifiers (objective, free after setup).
