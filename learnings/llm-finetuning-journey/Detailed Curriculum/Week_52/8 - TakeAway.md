# Week 52 TakeAway — Phase 5 Gate

**One-liner:** Phase 5 = REINFORCE → PPO → DPO → GRPO; each method solves one specific problem; verifiable rewards make GRPO the best for SQL.

---

## Phase 5 Method Progression

```
Week 41: REINFORCE → policy gradient theorem, baseline subtraction
Week 42: PPO     → clip + GAE + KL penalty; RLHF pipeline
Week 43: DPO     → closed-form KL-constrained RL; eliminates RL loop
Week 44: Data    → execution-based SQL preference labeling
Week 45: Apply DPO → v2: SFT + DPO
Week 46: GRPO   → group-relative advantage; no critic; verifiable rewards
Week 47: Design → multi-level SQL reward function; anti-hack guards
Week 48: Run    → v3: SFT + DPO + GRPO
Week 49: Zoo    → KTO, ORPO, SimPO; decision framework
Week 50-51: Iter → targeted experiments; pick best model
Week 52: Gate   → verify understanding and deliverables
```

---

## Three Key Derivations (memorize these)

**DPO loss (6 steps):**
```
1. max_π E[r] - β·KL(π||π_ref)
2. Optimal: π*(y|x) ∝ π_ref(y|x)·exp(r(x,y)/β)
3. Invert: r(x,y) = β·log(π*(y|x)/π_ref(y|x)) + β·log Z(x)
4. Z(x) cancels in r(y_w) - r(y_l)
5. Substitute into BT: L_BT = -log σ(r(y_w) - r(y_l))
6. L_DPO = -log σ(β·(log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))
```

**GRPO advantage:**
```
A_i = (r_i - mean({r_1,...,r_K})) / (std({r_1,...,r_K}) + ε)
All equal → A_i = 0 → zero gradient (correct behavior)
```

**Policy gradient (REINFORCE):**
```
∇J(θ) = E_{τ~π_θ}[Σ_t ∇ log π_θ(a_t|s_t) · G_t]
[log-derivative trick: ∇p = p·∇log p]
```

---

## Phase 5 Gate Checklist

- [ ] DPO derivation: can do on paper in 15 min
- [ ] PPO objective: can write from memory
- [ ] GRPO advantage: can state and explain
- [ ] v1, v2, v3 on HF Hub: all load and generate SQL
- [ ] v3 exec accuracy > v1 exec accuracy (quantified)
- [ ] GRPO W&B run exists with 500+ steps logged
- [ ] Phase 5 final eval report written
- [ ] Reflection document written

---

## Method Selection (final summary)

| Question | Answer |
|---|---|
| Have verifiable reward? | → GRPO |
| Have paired offline prefs? | → DPO |
| Have unpaired labels? | → KTO |
| Memory very constrained? | → ORPO or SimPO |
| Need maximum flexibility? | → PPO |

---

## Phase 6 Preview

- Goal: SFT → DPO → GRPO on 50K dataset; quantize; deploy; write paper
- Starting point: your Phase 5 best model as proof of concept
- First task (Week 53): Read LIMA and Tülu 3; design data quality strategy for 50K
