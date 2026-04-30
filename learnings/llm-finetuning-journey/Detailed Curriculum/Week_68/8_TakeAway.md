# Week 68 TakeAway — Technical Report: Training Pipeline Section

Every number in the training section must trace to a W&B config tab, not to memory.

## Key Section Structure

```
4. Training Pipeline
  4.1 Base model: name, params, why chosen, license (1 paragraph)
  4.2 CPT: loss formula, data, hyperparams, perplexity drop
  4.3 SFT: loss formula, data reference, hyperparams, val loss
  4.4 DPO: loss formula, beta, reward margin, accuracy gain over SFT
  4.5 GRPO: reward formula, K, steps, accuracy gain over DPO
  4.6 Compute: GPU-hours per stage, total, approximate cost, min hardware
```

## Loss Formulas to Include

```
L_CPT  = -E[log p_θ(x_t | x_{<t})]
L_SFT  = -E[log p_θ(y_t | x, y_{<t})]      # assistant turns only
L_DPO  = -E[log σ(β(log p_θ(y_w|x)/p_ref(y_w|x) - log p_θ(y_l|x)/p_ref(y_l|x)))]
r_GRPO = 1.0 if execute(y)==execute(y_gold), 0.5 if valid SQL, 0.0 otherwise
```

## Decision Rules

- If you ran multiple trial runs before final: report final only, mention failures in footnote
- If a hyperparameter was tuned: report the search range and why you chose the final value
- If alpha = 2 × rank: state this explicitly with the QLoRA citation
- GPU-hours = num_GPUs × wall_clock_hours — always compute and report both
- Effective batch size = batch_size × grad_accum × num_GPUs — report the effective value

## Numbers to Remember

- Hyperparameter table must cover: LR, schedule, warmup, batch size, grad accum, LoRA r/alpha/targets, max seq len, steps, optimizer, weight decay, clip
- DPO β typical range: 0.05–0.5; values outside this are unusual and need justification
- GRPO K typical range: 4–16; K=8 is the most common default
- LoRA rank sweep: usually r ∈ {16, 32, 64, 128}; document the sweep even if brief

## Red Flags

- Training section says "we used standard hyperparameters": non-reproducible — list every value
- DPO section reports reward margin but no accuracy improvement vs SFT: add the ablation number
- Compute section gives wall-clock hours without GPU type: meaningless for reproducibility
- LoRA rank not justified: add at least a 2-point ablation (r=32 vs r=64)
- Loss formula omitted: include it — it confirms you used the correct objective
