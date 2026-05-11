# Week 52 Gate — Reference Answers

## Gate Task 1 — Mathematical Derivations (Reference)

### Derivation 1: DPO Loss

```
Step 1 (KL-constrained RL objective):
  max_π E_{y~π}[r(x,y)] - β·KL(π||π_ref)

Step 2 (Expand KL into single expectation):
  = max_π E_{y~π}[r(x,y) - β·log(π(y|x)/π_ref(y|x))]

Step 3 (Optimal policy via functional optimization):
  The unique maximizer is:
  π*(y|x) = π_ref(y|x) · exp(r(x,y)/β) / Z(x)
  where Z(x) = Σ_y π_ref(y|x)·exp(r(x,y)/β)  [partition function]

Step 4 (Reparameterize r in terms of π*):
  r(x,y) = β·log(π*(y|x)/π_ref(y|x)) + β·log Z(x)

Step 5 (Compute preference difference, Z(x) cancels):
  r(x,y_w) - r(x,y_l) = β·log(π*(y_w|x)/π_ref(y_w|x)) - β·log(π*(y_l|x)/π_ref(y_l|x))
  [Z(x) cancels because it depends only on x, not y]

Step 6 (Bradley-Terry → DPO loss, replace π* with π_θ):
  L_DPO(π_θ) = -E_{(x,y_w,y_l)} [
    log σ(β·log(π_θ(y_w|x)/π_ref(y_w|x)) - β·log(π_θ(y_l|x)/π_ref(y_l|x)))
  ]
```

### Derivation 2: REINFORCE Policy Gradient

```
J(θ) = E_{τ~π_θ}[R(τ)] = Σ_τ p(τ|θ)·R(τ)

∇_θ J(θ) = Σ_τ ∇_θ p(τ|θ)·R(τ)
           = Σ_τ p(τ|θ)·[∇_θ log p(τ|θ)]·R(τ)   [log-derivative trick: ∇p = p·∇log p]
           = E_{τ~π_θ}[∇_θ log p(τ|θ)·R(τ)]

Since log p(τ|θ) = Σ_t log π_θ(a_t|s_t) + [transition terms that don't depend on θ]:
  ∇_θ J(θ) = E_{τ~π_θ}[Σ_t ∇_θ log π_θ(a_t|s_t)·G_t]

Valid for non-differentiable R because: R(τ) appears only as a scalar weight, not 
  differentiated. Only log π_θ(a_t|s_t) is differentiated — this is always differentiable.
```

### Derivation 3: GRPO Advantage and REINFORCE Connection

```
GRPO advantage: A_i = (r_i - mean({r_1,...,r_K})) / std({r_1,...,r_K})

Connection to REINFORCE baseline subtraction:
  REINFORCE with baseline: ∇J ≈ Σ_t (G_t - b(s_t)) · ∇ log π_θ(a_t|s_t)
  Any baseline b(s_t) that does not depend on the action is valid (zero bias).
  GRPO's baseline: b = mean(r) [the within-group mean reward].
  This is a state-independent (per-prompt) baseline — valid for the same reason.

When all K rewards equal (r_1 = r_2 = ... = r_K):
  mean(r) = r (all are equal), std(r) = 0
  A_i = (r_i - r) / 0 → 0/0 → handled by (std + ε) → A_i ≈ 0
  The gradient is 0 for all completions → no update.
  This is correct: if all completions are equally good (or bad), there is no evidence 
  about which direction to update the policy.

Why deterministic verifiable reward makes within-group mean a good baseline:
  For deterministic rewards: same (prompt, completion) → same reward always.
  With K=8, the within-group mean estimates the true expected reward E[r|prompt].
  As K→∞, the group mean → E[r|prompt] exactly (law of large numbers).
  For SQL: "does this SQL execute correctly?" is deterministic — same SQL, same DB, 
  same answer always. The K=8 mean is a reliable estimate of E[r|prompt].
  For stochastic rewards (different human raters give different scores), K=8 gives 
  a noisy estimate — this is why GRPO is less suited for human preference alignment.
```

---

## Gate Task 3 — GRPO Verification (What to Look For in W&B)

Your W&B run at `week-48-grpo-sql` should show:
- At least 500 steps logged (x-axis: 0 to 500+)
- `train/mean_reward` metric: starts near your diagnostic baseline (Week 47 Task 2 result) and trends upward
- `train/reward_std` metric: should be non-zero throughout healthy training
- `train/kl_divergence` metric: should stay below 10 nats

If W&B access is lost: paste the first 10 lines of the GRPO training log showing step count, mean_reward, and reward_std values.

---

## How to Verify You Passed the Gate

1. Written derivations: ask yourself "could I reproduce this on a whiteboard in a job interview?" If yes, you pass.
2. All three HF Hub models load and generate SQL.
3. The eval report shows at minimum: `phase5-best exec accuracy > v1 exec accuracy`.
4. Reflection: honest, specific, not generic platitudes.
5. Phase 6 preview shows you have read the master curriculum for Phase 6 and understand what is coming.
