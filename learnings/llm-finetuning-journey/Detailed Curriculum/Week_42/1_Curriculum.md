# Week 42 — PPO and the Original RLHF Stack

## Learning Objectives

By the end of this week, you will be able to:

- Derive the PPO clipping objective and explain why it replaces the raw policy gradient update
- Explain Generalized Advantage Estimation (GAE) and state its two hyperparameters (λ, γ)
- Describe the InstructGPT three-stage pipeline: SFT → reward model → RL fine-tuning
- Explain the role of the reference model and the KL penalty in RLHF
- Read and annotate TRL's PPOTrainer source code and identify where each component lives

## Why PPO?

REINFORCE from Week 41 has a fatal flaw for large models: unbounded policy updates. A large gradient step can move the policy so far from the data distribution that subsequent rollouts are incoherent, and the model can never recover. PPO (Proximal Policy Optimization) constrains each update so the new policy stays "close" to the old one. This is the dominant algorithm behind the original ChatGPT/InstructGPT training.

## Concepts

### The Trust Region Problem

When you update policy parameters θ, you want to improve J(θ) but not overshoot. The Trust Region Policy Optimization (TRPO) paper addressed this with a KL constraint:

```
maximize  L^{CPI}(θ)
subject to  E[KL(π_old || π_new)] ≤ δ
```

This is hard to optimize (second-order methods required). PPO approximates it with a clipped surrogate objective that is much simpler.

### PPO Clipping Objective

Define the probability ratio:

```
r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)
```

The unclipped surrogate is r_t(θ) · A_t. PPO clips this ratio to prevent large updates:

```
L^{CLIP}(θ) = E_t [ min( r_t(θ) · A_t,  clip(r_t(θ), 1-ε, 1+ε) · A_t ) ]
```

where ε is typically 0.1 or 0.2. The clipping creates a "pessimistic" bound: if the policy change would increase the objective but requires a large ratio (r_t >> 1), the clip prevents the optimizer from going too far. If the advantage is negative, the clip prevents the policy from reducing the probability of bad actions too aggressively.

In practice, you also subtract an entropy bonus (to encourage exploration) and a value function loss (to train the critic):

```
L^{PPO}(θ) = L^{CLIP}(θ) - c1 · L^{VF}(θ) + c2 · H[π_θ]
```

### Generalized Advantage Estimation (GAE)

Plain advantage estimation A_t = G_t − V(s_t) has high variance (G_t uses full Monte Carlo returns). GAE interpolates between 1-step TD error and full returns using λ:

```
δ_t = r_t + γ · V(s_{t+1}) - V(s_t)           # TD error
A_t^{GAE} = Σ_{k=0}^{∞} (γλ)^k δ_{t+k}
```

- λ = 0: A_t = δ_t (1-step TD, low variance, high bias)
- λ = 1: A_t = G_t − V(s_t) (full MC, low bias, high variance)
- λ = 0.95 is a common default that balances the two

GAE is an exponentially-weighted sum of TD errors. This is efficient to compute in one backward pass over the trajectory.

### The RLHF / InstructGPT Pipeline

The InstructGPT paper (Ouyang et al. 2022) established the three-stage RLHF pipeline that all subsequent RLHF work builds on:

**Stage 1 — Supervised Fine-Tuning (SFT):** Fine-tune the base model on a set of high-quality human demonstrations. This is exactly what you did in Phase 4 (your v1 model).

**Stage 2 — Reward Model (RM) Training:** Collect comparison data: for each prompt, show humans two completions and ask which they prefer. Train a reward model (typically the SFT model with a scalar output head) to predict human preferences. The loss is a Bradley-Terry preference model: maximize log σ(r_w − r_l) where r_w is the reward for the preferred completion and r_l for the rejected one.

**Stage 3 — RL Fine-Tuning with PPO:** Use PPO to optimize the SFT model against the reward model. The reward for a completion y given prompt x is:

```
r(x, y) = r_RM(x, y) - β · KL(π_θ(y|x) || π_SFT(y|x))
```

The KL term penalizes how far the policy drifts from the SFT model. Without it, the policy quickly learns to exploit the reward model (reward hacking) by generating text that scores high on the RM but is meaningless or degenerate.

### The Reference Model

The reference model π_ref (usually a frozen copy of the SFT model) serves two purposes:
1. It anchors the KL penalty: the trained model is penalized for diverging from it
2. In DPO (next week), it is used to compute the implicit reward

In TRL's PPOTrainer, the reference model is maintained as a separate frozen model. At each step, the KL divergence is computed by running both the training model and the reference model on the same input and comparing their log-probabilities.

### What Actually Breaks in RLHF with PPO

PPO-based RLHF has several practical failure modes you should know:

- **Reward hacking:** The model finds adversarial inputs that score high on the RM but are not actually good. Symptom: RM score increases but human preference ratings drop.
- **KL explosion:** If β is too small, the policy drifts too far from π_ref and generates incoherent text.
- **Value model instability:** The critic (value network) is often initialized from the SFT model; if it diverges, advantage estimates become noisy.
- **Long training runs are expensive:** Every PPO step requires forward passes through 4 models: actor, critic, reward model, and reference model.

This complexity is exactly why DPO (Week 43) was appealing — it eliminates the RL training loop entirely.

### Annotating TRL's PPOTrainer

The [TRL PPOTrainer source](https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py) is the reference implementation. Key functions to find and annotate:

- `step()`: the main training step; identify where rollouts are collected, where rewards are computed, where KL is added, and where PPO updates happen
- `compute_advantages()`: GAE calculation
- `loss()`: the PPO clipping loss
- `_generate_batched()`: how completions are generated from the current policy
- Where the reference model is called for KL computation

## Connections

Builds on: Week 41 (REINFORCE, policy gradient, advantage estimation).

Week 43 (DPO) directly replaces the RM + PPO stages with a single classification loss. Understanding what DPO replaces requires understanding what it is replacing.

Week 46 (GRPO) replaces the critic network (which PPO requires for GAE) with a group-relative baseline. You need to understand what the critic does in PPO to appreciate why removing it is non-trivial.

## Common Misconceptions

- "PPO and RLHF are the same thing." PPO is the RL algorithm. RLHF is the pipeline (SFT → RM → PPO). RLHF can in principle use other RL algorithms.
- "The KL penalty is for numerical stability." It is for alignment stability — preventing reward hacking and maintaining language coherence.
- "You need to run PPO to do alignment today." Most practitioners in 2024–2025 use DPO or GRPO. PPO is the foundation to understand, not the tool to reach for first.
- "The reference model is updated during training." It is frozen. If it were updated, the KL would always be zero.

## Time Allocation (6–8 hours)

- 2 hours: Read PPO paper (focus on sections 1–3). Read InstructGPT paper sections 3–4.
- 1 hour: Watch Yannic Kilcher's InstructGPT video.
- 30 min: Read Spinning Up GAE section.
- 2.5–3 hours: Read and annotate TRL PPOTrainer source code. Add inline comments explaining each step.
- 30 min: Write a 1-paragraph summary of what each stage of InstructGPT does and why.
