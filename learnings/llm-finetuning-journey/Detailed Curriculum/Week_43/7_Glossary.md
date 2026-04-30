# Week 43 Glossary

**DPO (Direct Preference Optimization)**: Alignment method that trains directly on preference pairs by reparameterizing the reward as a log-ratio; eliminates the RL training loop.

**KL-constrained RL objective**: max_π E[r] − β·KL(π||π_ref); the objective that both PPO (approximately) and DPO (exactly) optimize.

**Closed-form optimal policy**: π*(y|x) ∝ π_ref(y|x)·exp(r(x,y)/β); the exact solution to the KL-constrained RL objective.

**Partition function Z(x)**: Normalizing constant in the optimal policy: Z(x) = Σ_y π_ref(y|x)·exp(r(x,y)/β); cancels in the DPO loss.

**Log-ratio (DPO)**: log π_θ(y|x) − log π_ref(y|x); the implicit reward signal in DPO, measuring how much the trained model prefers y relative to the reference model.

**Reward margin**: `rewards/chosen − rewards/rejected` in TRL; the key health metric for DPO training — should be positive and growing.

**Bradley-Terry model**: Probabilistic model of pairwise preferences: P(y_w preferred) = σ(r_w − r_l); the preference model that DPO's loss is derived from.

**Offline alignment**: Training on a fixed dataset of preference pairs collected before training begins (DPO approach); contrast with online alignment (PPO/GRPO which generate fresh rollouts).

**Distributional shift (DPO)**: When the trained policy drifts far from the policy that generated the preference data, the offline data becomes stale and DPO's assumptions break down.

**β (DPO)**: The KL penalty coefficient; higher β keeps the model closer to π_ref; lower β allows more divergence.

**Reference model (DPO)**: Frozen SFT model (π_ref) used to compute log-ratios at every DPO training step.

**Preference pair**: A tuple (prompt x, chosen completion y_w, rejected completion y_l) forming one DPO training example.
