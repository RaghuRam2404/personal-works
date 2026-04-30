# Week 46 — GRPO and RLVR: The 2025 Breakthrough

## Learning Objectives

By the end of this week, you will be able to:

- Explain GRPO (Group Relative Policy Optimization) and why it eliminates the critic network
- State why verifiable rewards (RLVR) changed the alignment landscape in 2025
- Describe how DeepSeek-R1 applied GRPO at scale to produce reasoning behavior
- Read and annotate TRL's GRPOTrainer source code
- Write a 2-page explainer in your own words connecting GRPO theory to your SQL training plan

## Context: Why 2025 Changed Everything

In January 2025, DeepSeek released two papers in quick succession: DeepSeekMath (introducing GRPO as an algorithm) and DeepSeek-R1 (demonstrating that a model trained with GRPO + verifiable math rewards can match OpenAI o1 on reasoning benchmarks at a fraction of the cost). The key insight: if your reward function is a verifiable rule (does the math answer equal the expected answer? does the code pass the test?), you do not need a learned reward model. The verifier is ground truth.

For your SQL training: you have had a verifiable reward since Week 44. SQL execution is ground truth. This week you learn the algorithm that will let you use it most effectively.

## Concepts

### What Is Wrong with PPO for Verifiable Rewards?

PPO was designed for dense reward environments. For LLMs with verifiable rewards, the problems are:
1. You need a critic network (value function) that estimates V(s_t) at each token position. This critic is expensive and hard to train — it receives a signal only at the end of each episode (sparse).
2. The actor and critic together require 2× the memory of a single model.
3. GAE requires the critic to be accurate before it helps. Early in training, the critic is random noise, and GAE advantage estimates are useless.
4. Running the actor + critic + reward model + reference model for every training step is prohibitively expensive for 7B+ models.

GRPO solves problem 1 and 2 by replacing the critic entirely.

### GRPO: Group Relative Policy Optimization

GRPO (introduced in DeepSeekMath, Shao et al. 2024) generates K completions per prompt (a "group") and uses the within-group normalized rewards as the advantage estimate. No critic required.

**Algorithm:**

For each prompt x in the batch:
1. Sample K completions: {y_1, ..., y_K} from the current policy π_θ
2. Compute reward for each: {r_1, ..., r_K}
3. Compute group-relative advantage:
   ```
   A_i = (r_i - mean({r_1,...,r_K})) / std({r_1,...,r_K})
   ```
4. Optimize the clipped policy gradient loss (same PPO clip as Week 42) using A_i as the advantage

The group-relative normalization is the baseline subtraction from REINFORCE (Week 41) applied at the group level. Instead of a learned V(s), you use the empirical mean of the group's rewards as the baseline. This has a crucial property: if all K completions in a group receive the same reward, A_i = 0 for all of them, and there is no gradient. The gradient only flows when some completions are better than others within the group.

### Why No Critic?

The critic in PPO serves as a variance-reduction baseline. GRPO replaces it with the empirical within-group mean. This works well when:
- The reward function is deterministic (verifiable): the same prompt + same completion always gives the same reward. For SQL execution, this is true.
- K is large enough (K = 8 or 16 is typical) to get a reliable estimate of the mean reward.
- The reward function can be computed quickly in parallel.

For non-verifiable rewards (human preferences), the within-group mean is not a reliable baseline because the rewards are stochastic (different human raters give different scores). PPO is better there. For verifiable rewards (SQL, math, code), GRPO is significantly better because:
- No critic training instability
- Half the memory footprint
- Fresh rollouts every step (on-policy)

### The GRPO Loss

GRPO uses the same PPO clipping objective but with group-relative advantages:

```
L_GRPO(θ) = E_{(x, {y_i}, {r_i})} [
    (1/K) Σ_i min(
        (π_θ(y_i|x) / π_θ_old(y_i|x)) · A_i,
        clip(π_θ(y_i|x) / π_θ_old(y_i|x), 1-ε, 1+ε) · A_i
    )
    - β · KL(π_θ || π_ref)
]
```

This is PPO clip applied to each completion in the group, then averaged over the group. The KL penalty to the reference model is still present (same as RLHF).

### RLVR: Reinforcement Learning with Verifiable Rewards

RLVR is not a new algorithm — it is a design pattern: use RL training with a reward function that is a deterministic verifier. The verifier can be:
- A unit test runner (code)
- A SQL executor (your domain)
- A math checker (equals expected answer)
- A formal proof checker

The key property: the verifier provides a ground-truth signal that cannot be "hacked" the way a learned reward model can. DeepSeek-R1 showed that training with RLVR on math problems produces a model that develops reasoning chains spontaneously — the model discovers that "thinking step by step" improves its answers, and the reward signal reinforces this.

### DeepSeek-R1 and Chain-of-Thought Emergence

The surprising finding of DeepSeek-R1 (DeepSeek AI, 2025): when trained with GRPO on math verifiable rewards, the model spontaneously develops an "aha moment" behavior — it learns to pause, reconsider, and try different approaches. The reward signal is just "is the final answer correct?" — no supervision of the reasoning process. Yet the model develops structured reasoning chains.

For SQL: this implies that if you train with GRPO and a strong execution reward, your model may learn to check its own SQL before emitting it (e.g., generating a "let me verify the JOIN conditions" chain before the final query). This is a speculative benefit for your Week 47–48 training, but it is worth watching for.

### TRL GRPOTrainer Source

Key locations to find and annotate in [TRL's GRPOTrainer](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py):

- Where K completions are generated per prompt (`generate`)
- Where group-relative advantages are computed (look for `mean`, `std`, normalization)
- Where the PPO clip is applied (same as PPOTrainer)
- Where the reference model KL is computed and subtracted
- Where the reward function is called (`reward_model` or `reward_fn` argument)

## Connections

Builds on: Week 41 (REINFORCE + baseline subtraction — GRPO is a group baseline), Week 42 (PPO clipping — GRPO reuses the clip objective).

Week 47: The reward function you design this week conceptually, you will implement and run in Weeks 47–48.

Week 48: The actual GRPO training run on your SQL model.

## Common Misconceptions

- "GRPO is a simpler version of PPO." It is different, not simpler — the simplification (no critic) comes at the cost of needing more rollouts per prompt (K=8 vs K=1 in PPO).
- "RLVR only works for math." It works for any task with a deterministic verifier. SQL execution is a perfect fit.
- "DeepSeek-R1 uses GRPO exclusively." R1 combines GRPO with SFT on generated reasoning traces (cold start). The pure RL stage uses GRPO.
- "GRPO requires K distinct reward outcomes." It works even if all K get the same reward — the gradient is zero and the policy does not change, which is correct (no signal = no update).

## Time Allocation (6–8 hours)

- 2 hours: Read DeepSeekMath paper (Section 3 on GRPO) and DeepSeek-R1 paper (Sections 2–3).
- 1 hour: Read HuggingFace LLM Course Chapter 12 on implementing GRPO.
- 1 hour: Watch Karpathy's LLM Year in Review (the RLVR section) and Yannic Kilcher's DeepSeek-R1 video.
- 2–3 hours: Read and annotate TRL GRPOTrainer source.
- 30 min: Write your 2-page explainer in `week-46-grpo/grpo_explainer.md`.
