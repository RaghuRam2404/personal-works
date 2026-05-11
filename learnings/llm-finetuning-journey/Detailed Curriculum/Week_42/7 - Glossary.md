# Week 42 Glossary

**PPO (Proximal Policy Optimization)**: Policy gradient algorithm that clips the policy update ratio to prevent destabilizing large updates.

**Clipping objective**: PPO's core mechanism — the loss uses min(r·A, clip(r, 1-ε, 1+ε)·A), ensuring the policy does not change too much in one update.

**Probability ratio r_t(θ)**: The ratio π_θ(a|s) / π_old(a|s); measures how much the policy has changed for a specific action.

**GAE (Generalized Advantage Estimation)**: Exponentially-weighted sum of TD errors parameterized by λ; interpolates between 1-step TD (λ=0) and full Monte Carlo returns (λ=1).

**TD error (δ_t)**: r_t + γ·V(s_{t+1}) − V(s_t); the one-step temporal difference residual used in GAE.

**Critic (value network)**: Neural network that estimates V(s_t); trained jointly with the policy in PPO. Provides the baseline for advantage computation.

**Reference model (π_ref)**: Frozen copy of the SFT model; used to compute KL divergence penalty during RLHF training.

**KL penalty**: −β · KL(π_θ || π_ref) term added to the reward; prevents the policy from drifting too far from the SFT baseline.

**Reward model (RM)**: A language model fine-tuned to output a scalar score predicting human preference; trained on pairwise comparison data.

**Bradley-Terry model**: Probabilistic model used to train the reward model: P(y_w preferred) = σ(r_w − r_l).

**Reward hacking**: When the policy finds inputs or patterns that score highly on the RM but do not correspond to actual quality; a form of Goodhart's Law.

**RLHF (Reinforcement Learning from Human Feedback)**: Three-stage pipeline: SFT → reward model training → PPO fine-tuning.

**InstructGPT**: OpenAI's 2022 paper introducing the SFT → RM → PPO pipeline for aligning GPT-3 to follow instructions.

**Trust region**: A constraint on how much the policy is allowed to change per update; PPO approximates the TRPO trust region using clipping.

**PPO epochs**: Number of gradient steps taken on the same batch of rollouts; typically 1–4 in RLHF to avoid over-optimizing stale data.
