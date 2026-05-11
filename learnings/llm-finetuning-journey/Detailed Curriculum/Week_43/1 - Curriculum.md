# Week 43 — DPO: Skipping the Reward Model

## Learning Objectives

By the end of this week, you will be able to:

- Derive the DPO loss from the KL-constrained RL objective in Appendix A.1 of the DPO paper
- Explain why DPO produces the same optimal policy as PPO-RLHF without running a RL training loop
- Identify the assumptions DPO makes and when they break down
- Run a DPO training job on `HuggingFaceH4/ultrafeedback_binarized` using TRL's DPO trainer
- Explain the relationship between the DPO loss and the Bradley-Terry reward model

## The Core Insight

DPO (Direct Preference Optimization) is not a simplification of RLHF — it is a reparameterization. The RLHF objective has a closed-form optimal policy. DPO rewrites the reward model in terms of that optimal policy, then substitutes back into the preference loss, yielding a loss function that trains the policy directly on preference data. No reward model, no RL training loop, no value function.

This is the reason DPO dominated alignment training in 2024 and why understanding the math (not just the recipe) is essential.

## Concepts

### The KL-Constrained RL Objective

The RLHF objective is:

```
max_{π} E_{x~D, y~π} [r(x, y)] - β · KL(π || π_ref)
```

This is maximizing expected reward while staying close to the reference policy π_ref. This objective appears in PPO as an add-on (the KL penalty term), but it is actually the fundamental objective that PPO is approximately solving.

### The Closed-Form Optimal Policy

The remarkable result from the DPO paper (Appendix A.1): the optimal policy for this objective has a closed form:

```
π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp(r(x,y) / β)
```

where Z(x) = Σ_y π_ref(y|x) exp(r(x,y)/β) is the normalizing partition function (intractable to compute directly, but this does not matter — it will cancel).

This means: the optimal policy upweights sequences with high reward relative to the reference policy. The more the reward exceeds the average (as measured by Z), the higher the probability.

### Reparameterizing the Reward

Inverting the optimal policy equation to express the reward in terms of the policy:

```
r(x, y) = β · log(π*(y|x) / π_ref(y|x)) + β · log Z(x)
```

This says: the implicit reward of a completion under the optimal policy is proportional to its log ratio relative to the reference model. The partition function Z(x) drops out when you compute reward differences (chosen minus rejected), because Z(x) depends only on x, not y.

### The DPO Loss

Substituting this reparameterized reward into the Bradley-Terry preference loss:

```
P(y_w preferred over y_l | x) = σ(r(x, y_w) - r(x, y_l))
```

The DPO loss (for maximizing log-likelihood of preferences) becomes:

```
L_DPO(π_θ; π_ref) = -E_{(x, y_w, y_l) ~ D} [
    log σ(
        β · log(π_θ(y_w|x) / π_ref(y_w|x))
        - β · log(π_θ(y_l|x) / π_ref(y_l|x))
    )
]
```

This is a binary cross-entropy loss. You are training the model to predict the correct preference label, but the "features" are log-probability ratios between the training model and a frozen reference model.

Written compactly: `L_DPO = -log σ(β(log_ratio_w - log_ratio_l))` where `log_ratio_y = log π_θ(y|x) - log π_ref(y|x)`.

### Why DPO Eliminated PPO in Most Academic Work

PPO requires:
- 4 forward passes per step (actor, critic, RM, reference)
- Careful KL tuning, value function training, rollout collection
- Online generation (the policy must sample new completions during training)

DPO requires:
- 2 forward passes per step (training model, reference model)
- A fixed offline preference dataset — no generation during training
- One hyperparameter β (analogous to KL coefficient)
- Standard cross-entropy training infrastructure

DPO converges in hours on a single GPU for 7B models. PPO takes days and requires careful engineering.

### When DPO Breaks Down

DPO is not strictly better than PPO. Its assumptions are:

1. **Offline data assumption:** The preference data was collected once and is fixed. If the policy shifts significantly from the policy that generated the data, DPO becomes stale (distributional shift). PPO avoids this by generating fresh rollouts.

2. **No verifiable rewards:** DPO requires human (or AI) preference labels. If your reward is verifiable (SQL executes correctly), you do not need human labels and can generate fresh execution-based rewards. This is why GRPO (Week 46) is better for SQL.

3. **Implicit reward accuracy:** DPO assumes the Bradley-Terry model is correct. If human preferences are inconsistent or noisy, the implicit reward is corrupted.

A practical failure mode you will encounter in Week 45: if your DPO model starts generating more refusals ("I cannot generate SQL for that") despite having higher accuracy on chosen responses. This happens because the model learns to be conservative — lowering the probability of "rejected" completions also lowers the probability of completing the response at all if the refusal pattern is similar to rejection patterns.

### The DPO Gradient Intuition

The gradient of L_DPO with respect to θ pushes the model to:
- Increase log π_θ(y_w|x) (make chosen completions more likely)
- Decrease log π_θ(y_l|x) (make rejected completions less likely)

But it does this weighted by how "surprised" the current model is — the σ term is close to 0 when the model already correctly ranks the pair, so updates are small for easy pairs and large for hard pairs. This is exactly analogous to how cross-entropy loss works harder on the uncertain examples.

## Connections

Builds on: Week 42 (KL-constrained RL objective, Bradley-Terry reward model training, reference model concept).

Week 44 needs: understanding what a "preference pair" is and why execution-based labeling creates cleaner pairs than human feedback.

Week 45 needs: practical DPO training mechanics — β tuning, batch size, learning rate.

Week 46 (GRPO) partially replaces DPO for verifiable rewards — you need to understand DPO's limitations to motivate GRPO.

## Common Misconceptions

- "DPO does not use RL at all." DPO is derived from RL. It skips the RL training procedure but not the RL objective.
- "DPO replaces the reward model entirely." DPO replaces the explicit reward model at training time. The reward model is implicitly encoded in the preference data labels.
- "Higher β means less regularization." Higher β means more regularization — a larger KL penalty coefficient keeps the policy closer to π_ref.
- "DPO is always better than PPO." For verifiable rewards (code execution, math checking), PPO or GRPO with a fresh reward signal is often better.

## Time Allocation (6–8 hours)

- 2 hours: Read DPO paper fully, including Appendix A.1 derivation. Take notes on each step.
- 30 min: Re-read Week 42 notes on Bradley-Terry loss to connect the notation.
- 1 hour: Watch Umar Jamil's DPO explanation video (linked in Resources).
- 30 min: Read TRL DPO Trainer docs.
- 2.5–3 hours: Run philschmid's DPO notebook end-to-end on ultrafeedback_binarized.
- 30 min: Derive the DPO loss on paper without looking at notes. This is the acceptance criterion.
