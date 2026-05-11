# Week 30 Quiz Answers

## Q1 — Answer: B

**Answer:** B. 16 × (2048 + 2048) = 65,536

**Why:** Matrix A has shape (r × d_in) = (16 × 2048) = 32,768 parameters. Matrix B has shape (d_out × r) = (2048 × 16) = 32,768 parameters. Total = 65,536. The formula is r × (d_in + d_out).

**Why others are wrong:**
- A: This only counts lora_A, forgetting lora_B.
- C: This is full fine-tuning parameter count, not LoRA.
- D: This uses alpha (32) instead of rank (16) — alpha is a scaling hyperparameter, not a dimension.

---

## Q2 — Answer: B

**Answer:** B. Zero initialization ensures the adapter starts as a no-op.

**Why:** At initialization, delta_W = B @ A = 0 × A = 0. Therefore the LoRA model outputs exactly the same as the pretrained model for any input. This is critical for stable training: if the adapter started with a random non-zero delta_W, the model would immediately diverge from its pretrained state before it has learned anything. The zero initialization means training starts from the pretrained model's equilibrium, which is what we want.

**Why others are wrong:**
- A: Gradient explosion is controlled by gradient clipping and learning rate, not weight initialization.
- C: Gradients flow through B from step 1 regardless of initialization; "learning first" is not a meaningful distinction.
- D: Layer norm epsilon prevents division by zero; unrelated to LoRA initialization.

---

## Q3 — Answer: C

**Answer:** C. Merge B @ A into W and discard.

**Why:** The merge operation: W_merged = W + (B @ A) × (alpha / r). After merging, the forward pass is just y = W_merged x — a single matrix multiply with no extra LoRA overhead. This gives you the fine-tuned model's behavior with zero inference cost increase compared to the base model. You cannot do this with full SFT (W is already updated in-place), but with LoRA it is a free optimization.

---

## Q4 — Answer: B

**Answer:** B. Decouples learning rate sensitivity from rank.

**Why:** Without the scaling factor, changing r from 8 to 16 (doubling the adapter capacity) would also double the magnitude of the LoRA output at any given gradient step, effectively changing the update scale. The alpha/r term compensates: if alpha stays fixed and r doubles, the scaling halves, keeping the overall update magnitude approximately constant. This means you can sweep r independently of alpha without needing to re-tune the learning rate.

---

## Q5 — Answer: B

**Answer:** B. All linear layers — more directions to improve.

**Why:** The empirical consensus from Sebastian Raschka's analysis and other experiments is that covering all linear layers consistently outperforms covering only attention projections at the same rank, on supervised fine-tuning tasks. The MLP layers process much of the model's "knowledge" — restricting LoRA to only attention layers leaves the MLP's representation untouched. At rank 8, total parameter count for all layers is still less than 1% of 7B parameters, so overfitting on 5K examples is not the concern with rank 8.

**Why A is wrong:** On a 5K dataset, the total parameter count difference (all layers vs. q+v only) is still tiny — both are well under 100M trainable params. The extra capacity from all layers helps, not hurts.

---

## Q6 — Short Answer

Matrix A has shape (r, d_in): it projects the input from d_in dimensions down to r dimensions. Number of parameters = r × d_in.

Matrix B has shape (d_out, r): it projects from r dimensions back up to d_out. Number of parameters = d_out × r.

Total = r × d_in + r × d_out = r × (d_in + d_out).

The rank r is the "bottleneck" dimension — the number of independent directions the adapter can express. Since r << min(d_in, d_out) for typical LoRA configurations, the parameter count is dominated by the (d_in + d_out) factor rather than the quadratic d_in × d_out of the full matrix.

---

## Q7 — Short Answer

The colleague is partially correct but the practical implication is wrong. In theory, if the true optimal delta_W has rank 50, a rank-16 approximation will not reach the global optimum. However:

1. Empirically, delta_W from full fine-tuning experiments on instruction/task data tends to have effective rank much lower than 50 — often below 16 for moderate-scale tasks.
2. LoRA at rank 16 across all layers covers more "directions" than rank 16 on just a few layers, so the effective expressive capacity is higher than the per-layer rank suggests.
3. On small datasets (5K–15K examples), rank 64 LoRA frequently overfits compared to rank 16 LoRA — the true optimal delta_W in the data-limited regime may itself be low-rank.

Conclusion: the theoretical argument is valid but the practical threshold where it matters is much higher than rank 16 for typical fine-tuning tasks on 7B models.

---

## Q8 — Short Answer

The val loss of 0.3 with rank 64 on 5K examples is severe overfitting. Rank 64 adds approximately 8x more trainable parameters than rank 8, and on 5K examples the optimizer has driven the adapter to memorize training examples rather than learn generalizable SQL patterns.

Try rank 8 or rank 16. Also: verify that you are running early stopping based on val loss rather than training to fixed epoch count. Alternatively, keep rank 64 but reduce learning rate by 4x and add dropout (lora_dropout=0.05).

---

## Q9 — Scenario Answer

If both A and B are initialized randomly, delta_W = B @ A is a non-zero random matrix at step 0. For a 7B model, this means the model immediately changes its behavior from the pretrained state in a random direction — before any gradient information has been seen.

Concretely, suppose the pretrained model has loss 1.8 on the training data. With random A and B initialization, the initial loss might be 5.0 or 15.0 (random perturbation to all linear layer outputs). The first 5 training steps are now wasted recovering from this random initialization, not learning task-specific patterns.

The zero initialization of B guarantees that the model starts from the pretrained loss (1.8 in this example) and improves monotonically. The first gradient step updates B in the direction that reduces loss from the pretrained baseline. This is strictly better: you never pay the cost of recovering from random initialization.

The formal guarantee: at step 0, for any input x, LoRA output with B=0 equals the pretrained model output. Therefore, the initial loss equals the pretrained model's loss on your data, which is the natural starting point for fine-tuning.
