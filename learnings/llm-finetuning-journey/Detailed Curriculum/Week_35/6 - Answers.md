# Week 35 Quiz Answers

## Q1 — Answer: B

**Answer:** B. 2e-4 — empirical consensus for LoRA.

**Why:** 2e-4 is the most commonly validated starting LR for LoRA fine-tuning of 7B models across dozens of published experiments (Raschka's analysis, QLoRA paper, Alpaca/Vicuna training). The LoRA scaling factor `alpha/r` already modulates the effective update magnitude — so the optimal LR is higher than full SFT (where 1e-5 is typical) but lower than pretraining. 2e-4 with cosine decay and 5% warmup is the safe starting point before any sweeping.

---

## Q2 — Answer: B

**Answer:** B. Run B — same quality, 3x faster, less risk.

**Why:** If eval loss is equivalent, the run with fewer epochs and higher LR provides the same model quality with less computational cost and less overfitting risk. More epochs on the same data increase the chance of memorizing the training set. In production, you want the fastest path to equivalent quality — Run B achieves that. The "more epochs = more thorough" intuition applies in pretraining (more data seen) but not in SFT (same data repeated).

---

## Q3 — Answer: B

**Answer:** B. Multiply LR by 2 (square root scaling).

**Why:** The linear scaling rule (multiply LR linearly with batch size) is derived for convex optimization and often over-shoots for neural networks. The square root scaling rule (Goyal et al.) is more conservative: LR_new = LR_old × sqrt(batch_new / batch_old) = 2e-4 × sqrt(4) = 4e-4. For LLM fine-tuning, even square root scaling is aggressive — many practitioners simply keep the same LR when increasing batch size via gradient accumulation (since the gradient estimate is averaged, not summed, in PyTorch's default).

---

## Q4 — Answer: B

**Answer:** B. Linear LR warmup prevents gradient explosions during initialization.

**Why:** At the start of LoRA training, lora_B = 0 and lora_A is random. If the LR is immediately at 2e-4, the first gradient steps update B from zero based on the full gradient — these initial gradients can be large and cause the model to jump to a poor region of the loss landscape. Warmup linearly increases LR from ~0 to 2e-4 over the first 10% of steps, allowing B to stabilize gradually before taking large gradient steps. 5–10% warmup is standard for SFT.

---

## Q5 — Answer: B

**Answer:** B. If the SQL task requires complex multi-step queries needing higher expressiveness.

**Why:** On 1K examples, both ranks have insufficient data to demonstrate their capacity difference — the dataset is too small to distinguish rank 16 from rank 32 capability. With 15K diverse SQL examples including multi-join queries, window functions, CTEs, and subqueries, the higher capacity of rank 32 may allow the model to express more complex SQL transformation patterns. The right experiment: train both ranks on the full 15K and compare eval loss + execution correctness on your held-out test set.

---

## Q6 — Short Answer

With `alpha = 2 × rank`:
- Rank 16, alpha 32: scaling = 32/16 = 2.0. Effective LoRA output magnitude: lora_out × 2.0
- Rank 64, alpha 128: scaling = 128/64 = 2.0. Same scaling.

The scaling is the same — so the LoRA output magnitude is unchanged. However, rank 64 has 4× more trainable parameters than rank 16. Each gradient step can adjust 4× more independent directions. The LR=2e-4 gradient step moves each A and B parameter by the same amount per step, but rank 64 has more parameters to adjust — more total parameter movement per step.

To maintain the same "effective update magnitude per parameter": keep LR the same if the total parameter volume is the guide (which is what `alpha/rank` controls). But to account for the risk of larger total updates, reduce LR slightly when dramatically increasing rank: try LR = 1e-4 for rank 64 vs. 2e-4 for rank 16.

---

## Q7 — Short Answer

What happened: the model trained well for 1–2 epochs (train loss down to 0.4, eval loss down to 0.8 — both improving). Then in epoch 3, train loss continued falling but eval loss rose from 0.8 to 1.2. This is textbook overfitting: the model started memorizing training examples beyond what generalizes to the eval set.

Three changes for the next run:
1. **Enable early stopping:** `load_best_model_at_end=True`, stop training when eval loss stops improving for 2–3 consecutive evaluation checkpoints. This would have stopped training at the 0.8 eval loss checkpoint.
2. **Reduce epochs from 3 to 2:** On 10K examples, 2 epochs typically reaches the generalization peak before overfitting.
3. **Add lora_dropout=0.05:** Light regularization on the LoRA path helps prevent memorization on larger datasets. (If using Unsloth, the speed cost of dropout is small.)

---

## Q8 — Short Answer

The experiment:
- Run A: all linear layers (q,k,v,o,gate,up,down) with rank 32, alpha 64 → ~80M trainable params on 7B
- Run B: only q_proj and v_proj with rank 128, alpha 256 → 2 layers × 128 × (4096+4096) = ~134M trainable params on 7B (more params per layer but fewer layers)

Wait — the parameterization may not be equivalent. Recalculate:
- All 7 layers at rank 32: 7 × 32 × (d_in + d_out) × 28 layers ≈ 90M
- q+v at rank 128: 2 × 128 × (4096+4096) × 28 = 59M

Adjust: all layers at rank 32 has more total parameters. For a fair comparison, match total parameter count.

Run both configurations, train on 5K SQL examples, 2 epochs, measure eval loss and execution correctness on 100 held-out examples. My prediction: covering all linear layers at lower rank outperforms attention-only at higher rank because MLP layers store SQL syntax patterns essential for structured output.

---

## Q9 — Scenario Answer

3 hours of A100 time at 3 steps/second, 1K training subset, 2 epochs:
- 1K examples, packing ratio 3 → ~333 sequences → 333/16 ≈ 21 optimizer steps/epoch → 42 steps/run
- Each run: ~14 seconds of training + overhead = ~2–3 minutes per run
- Budget: 180 minutes / 3 minutes per run ≈ 60 runs theoretically; realistically 20–30 after startup overhead

**Strategy:**

Round 1 (LR sweep, 3 runs, 10 min): Fix rank=16, alpha=32. Sweep LR ∈ {1e-5, 1e-4, 2e-4}. → Identify best LR.

Round 2 (rank sweep, 3 runs, 10 min): Fix LR=best. Sweep rank ∈ {8, 16, 32}. → Identify best rank.

Round 3 (epoch sweep, 2 runs, 10 min): Fix LR and rank. Train 1 epoch vs. 3 epochs with early stopping. → Confirm epoch count.

Round 4 (verification, 1 run on 5K, 20 min): Re-run best config on 5K examples. Confirm results generalize.

Total: ~50 minutes. The remaining 2+ hours serve as buffer for debugging and re-runs. This structured sequential sweep is more informative than 30 random grid runs and respects the 3-hour budget.
