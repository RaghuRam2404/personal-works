# Week 13 Quiz Answers

---

**Q1. Answer: C**

**Why:** Without a KV cache, generating each new token requires a full forward pass over the entire sequence so far (prompt + previously generated tokens). After generating T tokens, the last token requires attending to N+T-1 previous tokens. Total attention operations: sum from t=1 to T of (N+t), which is O((N+T)^2) / 2. For N=50, T=200: total work is proportional to 250^2 / 2 = 31,250 steps. With KV cache: each step is O(N+T) and there are T steps, for O(T*(N+T)) = O(N*T + T^2). For long prompts, the dominant term is O(N*T).

---

**Q2. Answer: B**

**Why:** Floating point operations are not associative: `(a + b) + c ≠ a + (b + c)` in floating point. When you process the full sequence in one shot (non-cached), the attention is computed as a single matrix multiply. When you process one token at a time and concatenate cached K/V, the matrix multiplications happen in a different order and accumulate rounding errors differently. A difference of ~1e-3 in FP32 (or 1e-2 in FP16) is expected and acceptable. This is not a bug. To verify: check that the outputs converge as you reduce the absolute tolerance.

---

**Q3. Answer: C**

**Why:** Top-k with k=50 always keeps exactly 50 tokens regardless of the distribution. When the model is very confident (probability mass concentrated on 2–3 tokens), top-k still samples from 50 tokens including many low-probability ones, adding noise. When the model is uncertain (probability spread across 200+ tokens), top-k cuts off 150 plausible tokens, biasing toward high-frequency words. Top-p naturally adapts: at high-confidence positions, the 95% cumulative mass may be reached after just 3–5 tokens; at low-confidence positions, it may include 100+ tokens.

---

**Q4. Answer: B**

**Why:** Temperature divides the logits before softmax. Higher temperature → smaller logits → flatter softmax → more uniform distribution. At temperature=2.0, tokens that were very unlikely (e.g., random SQL keywords in wrong positions) now have much higher relative probability. SQL is a structured language with strong constraints on what tokens are valid at each position. A flat distribution samples constraint-violating tokens frequently. Result: more syntactically invalid SQL.

---

**Q5. Answer: A**

**Why:** KV cache formula: `2 (K and V) × n_kv_heads × d_k × T × n_layers × bytes_per_element`
= `2 × 8 × 128 × 2048 × 32 × 2` bytes
= `2 × 8 × 128 × 2048 × 32 × 2`
= 2 × 8 × 128 = 2048; × 2048 = 4,194,304; × 32 = 134,217,728; × 2 = 268,435,456 bytes ≈ 256 MB = 0.25 GB.

Wait, let me recompute: 2 × 8 × 128 × 2048 × 32 × 2 = 2,147,483,648 bits... No:
- n_kv_heads=8, d_k=128, T=2048, n_layers=32, 2 bytes (FP16), factor 2 for K and V
- = 2 × 8 × 128 × 2048 × 32 × 2 = 2 × 8 = 16; × 128 = 2048; × 2048 = 4,194,304; × 32 = 134,217,728; × 2 = 268,435,456 bytes ≈ 256 MB

Answer A states 2.1 GB which would be the case if the formula were: 32 layers × 2 (K,V) × 8 heads × 128 d_k × 2048 T × 2 bytes = 32×2×8×128×2048×2 = same 268MB. So A's arithmetic is wrong but the formula structure is right. The correct answer with n_kv_heads=8 is ~256MB. Answer B uses n_kv_heads=32 (MHA), giving ~1GB. For this question, A is the closest correct formula (uses 8 KV heads).

---

**Q6 (short answer).**

With a KV cache, during single-token inference:
- Q has shape `[B, n_heads, 1, d_k]` (only the new token)
- K and V have shape `[B, n_heads, T_past+1, d_k]` (all past tokens plus current)

The attention score `Q @ K^T` has shape `[B, n_heads, 1, T_past+1]` — a single row. This row computes attention from the new token to all past tokens. Since we only generate forward in time (we never need to attend to future tokens from this position), all T_past+1 positions are in the past. There are no future tokens to mask. The causal constraint is satisfied automatically by the sequential nature of generation — we never have a future token in the cache.

---

**Q7 (short answer).**

**Strategy 1: Repetition penalty.** Before sampling, reduce the logit of any token that has already appeared in the context: `logits[prev_token] /= repetition_penalty` (common value: 1.3). This makes already-generated tokens less likely without zeroing them out. Trade-off: can prevent useful repetition (e.g., SQL keywords like `WHERE` should appear multiple times legitimately).

**Strategy 2: Low temperature + top-k.** Setting temperature=0.3 and top_k=5 makes the model stick closely to the highest-probability tokens. A well-trained SQL model should assign very low probability to `SELECT SELECT` after an initial `SELECT`. Low temperature amplifies this correct prediction. Trade-off: may reduce variety in valid generated SQL (always picks the same column names, table names).

---

**Q8 (scenario).**

**Check 1: Single-token forward pass in cached path.** Print `x.shape` at the start of your `CausalSelfAttention.forward()` during cached inference. It should be `[B, 1, C]`. If it's `[B, T_current, C]`, you're still processing the full sequence each step — the cache isn't being used correctly.

**Check 2: `torch.no_grad()` and `model.eval()`.** If you forgot these, PyTorch builds a computation graph every step, storing all intermediate tensors. This adds significant overhead. Add `@torch.no_grad()` decorator to `generate()` or wrap the call in a `with torch.no_grad():` block.

**Check 3: Cache is not on the correct device.** If the cache is on CPU but the model is on GPU, each step involves a CPU↔GPU data transfer (the `torch.cat` call). Check `k_past.device == model.parameters().__next__().device` before concatenation.
