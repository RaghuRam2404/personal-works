# Week 15 Quiz Answers

---

**Q1. Answer: B**

**Why:** The mathematical intent of gradient accumulation is: `gradient_logical_batch = sum(gradient_micro_batch_i) / grad_accum_steps`. If you don't divide by `grad_accum_steps`, the accumulated gradient is `grad_accum_steps` times larger than intended. The effective LR is then `grad_accum_steps × intended_LR`. For `grad_accum_steps=32` and `max_lr=6e-4`, the effective LR becomes `1.92e-2` — far too high. The optimizer will take enormous steps, likely causing loss divergence (NaN) or severe oscillation in the first few steps.

---

**Q2. Answer: B**

**Why:** Adam maintains running estimates of the first moment (mean) `m_t` and second moment (variance) `v_t` of the gradients. These accumulate tiny values over thousands of steps. BF16 has only 7 bits of mantissa — it cannot represent values smaller than about `2^-7 ≈ 0.008` relative to the current magnitude. Small gradient updates would be rounded to zero, corrupting the moment estimates. This is why optimizer states are kept in FP32: they need the full 23-bit mantissa to accumulate updates accurately across 19,000+ steps.

---

**Q3. Answer: A**

**Why:** `F.scaled_dot_product_attention` uses Flash Attention when the hardware supports it (CUDA with Ampere+). Flash Attention tiles the computation to avoid writing the full `[T×T]` attention weight matrix to HBM. For T=1024, this is a `[B×H×1024×1024]` matrix — significant memory and bandwidth. You also no longer need to compute and register the `torch.tril` causal mask as a buffer: `is_causal=True` tells Flash Attention to apply the causal constraint internally without materializing the mask.

---

**Q4. Answer: B**

**Why:** Gradient clipping (`clip_grad_norm_ = 1.0`) is intended to handle occasional gradient spikes, not to be the permanent operating mode. If `grad_norm` consistently equals 1.0 at the clip limit, it means the raw gradients are consistently larger than 1.0 norm — the model is effectively operating at a lower LR than you set (because the clipped gradient is not proportional to the raw gradient magnitude). The most common cause is an LR that is too high for the current training dynamics. Try reducing `max_lr` by 20–30% and see if grad_norm drops to 0.3–0.8 (a healthy range).

---

**Q5. Answer: D**

**Why:** A) 45% after the same steps means the model is learning more efficiently — either it has more parameters (more expressiveness) or higher-quality data (more signal per token). Both are legitimate. C) If there's a bug where the model sees the correct answer label — for example, if the HellaSwag evaluation accidentally conditions on the answer token — the model trivially achieves near-100% accuracy. In practice, HellaSwag accuracy at 45% after 124M-equivalent training should raise suspicion of option C. Always sanity-check your eval by testing a randomly initialized model (should be ~25%) and a model that always picks candidate 0 (should be ~25%).

---

**Q6 (short answer).**

**Problem:** Large batch sizes are more stable and enable higher learning rates (linear scaling rule), but large batches require proportionally more GPU memory. A batch of 512k tokens won't fit in GPU memory in a single forward pass.

**How it works:** Instead of one large forward pass, run `G` micro-steps each with batch size `B × T / G`. The loss computed at each micro-step contributes a partial gradient: `loss.backward()` adds to the gradient buffer without zeroing it. After all G micro-steps, the gradient buffer contains the sum of all micro-step gradients — mathematically identical to the gradient from a single large batch.

**Normalization:** Without normalization, the accumulated gradient would be G times larger than a single-batch gradient. Before calling `loss.backward()`, divide the loss by `grad_accum_steps`: `(loss / G).backward()`. This ensures that `sum_i(loss_i / G) = mean(loss_i)` — the gradient of the mean over the full batch, which is what the optimizer expects.

---

**Q7 (short answer).**

With `B=16, T=1024, total_batch=524288`:
`grad_accum_steps = 524288 / (16 × 1024) = 32 micro-steps`

Switching to `T=2048`:
`grad_accum_steps = 524288 / (16 × 2048) = 16 micro-steps`

Half as many micro-steps per logical batch — good for throughput. But each micro-batch is twice as long. Memory implication: attention activations grow quadratically with T. For T=1024 → T=2048, the attention activation memory increases by 4x (since `T×T` quadruples). You will likely need to reduce batch size B from 16 to 8 to avoid OOM, restoring `grad_accum_steps` to 32. With Flash Attention, the attention memory is O(T) not O(T²), so the increase is only 2x — this is a key benefit of Flash Attention for long-context training.

---

**Q8 (scenario).**

`train_loss=3.2` at step 0 means the model is predicting the vocabulary near-uniformly — but with some preference, not fully random. The expected loss at step 0 for a randomly initialized model over a 50,257-token vocabulary is `-log(1/50257) ≈ 10.82`.

A loss of 3.2 corresponds to `e^(-3.2) ≈ 0.04` probability on the correct token — far above random. This means the model has already learned something before step 0, which is impossible unless:

**Most likely bug:** You accidentally loaded pretrained HuggingFace GPT-2 weights into your model at the start (perhaps via `from_pretrained`) and are evaluating the pre-loaded model, not a randomly initialized one. Check that you call `GPT(GPTConfig(...))` with fresh random initialization, not `GPT.from_pretrained(...)`, before the training run.

Alternative: Your training data is accidentally the same few repeated examples, and the model has trivially memorized them from a few gradient updates (if you accidentally ran an optimizer step before computing step-0 loss).

Fix: Ensure step-0 loss is evaluated before any `optimizer.step()` call, and verify model parameters are freshly random: `model.apply(init_weights_normal)`.
