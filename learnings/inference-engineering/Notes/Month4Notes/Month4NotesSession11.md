[← Month 4 notes](<../Month4Notes.md>) · [← Session 10](<Month4NotesSession10.md>)

## Session 11 — Part 1: Raschka single-head walkthrough

Code: [7 sr singleheadattention.py](../../Month%204/7%20sr%20singleheadattention.py). Sentence: "Life is short, eat dessert first".

**Meaning of each step**

| Step | Code | Shape | Meaning |
|---|---|---|---|
| Embed | `embed_layer(encoded)` | `[6, 16]` | One vector per token. Carries no information about the other tokens yet. |
| Project | `query = embed @ W_query.T` (likewise key, value) | `[6, 24]`, `[6, 24]`, `[6, 28]` | The same embedding is viewed in three roles. Query: what this token is looking for. Key: what this token advertises so it can be found. Value: the payload it hands over when selected. |
| Score | `omega = query @ key.T` | `[6, 6]` | `omega[i, j] = query[i] · key[j]`, a *learned* similarity (similarity after projection, not raw embedding similarity). It is a score, not the output. |
| Scale + softmax | `softmax(omega / sqrt(dk), dim=1)` | `[6, 6]` | Turns each row into a distribution over key positions: how the token `i` spends its attention budget. Each row sums to 1. Softmax runs over the key axis because the question is "which keys?" |
| Mix values | `attention_weights @ value` | `[6, 28]` | `context[i] = Σ_j w[i,j] · value[j]`. A weighted average of value vectors, giving a new, context-enriched vector for every token. |

One-line summary: query-key decides *where* to look, value decides *what* to take, and the context vector is the token's embedding rewritten using what it looked at.

**Convention rewrite** (article is column-vector, this curriculum is row-vector)

```text
Article:     q = W_q @ x        W_q: [d_q, d]      (code: embed @ W_query.T)
Curriculum:  q = x @ Wq         Wq:  [C, D]        (Wq = W_query.T)
```

**Differences from this curriculum**

- No causal mask. The `[6, 6]` matrix is full, so token 1 attends to tokens 2 to 5 as well. This is the encoder (BERT-style) special case. Causal attention masks `j > i` before the softmax.
- Unbatched `[T, d]` rather than `[B, T, C]`.
- Weights are `torch.rand` (all positive), not zero-mean.

**Experiments and findings**

1. **Shapes.** `W_query/W_key: [24, 16]`, `W_value: [28, 16]`; `query/key: [6, 24]`, `value: [6, 28]`; `omega` and weights `[6, 6]`; context `[6, 28]`. `dv != dk` works because values never enter the dot product; only `dq == dk` is required.
2. **First run did not match Raschka, even though the embeddings were identical** (`nn.Embedding(6, 16)`, seed 123). The cause was the weight initialization: I used `torch.randn` (normal, mean 0, std 1) and he uses `torch.rand` (uniform on [0, 1)). The same seed with a different sampler gives different numbers. After switching to `torch.rand` the outputs match his exactly.
3. **Reproduced Raschka's numbers.** `omega[1] = [8.5808, -7.6597, 3.2558, 1.0395, 11.1466, -0.4800]`, and `omega_24 = query[1] · key[4] = 11.1466`. Weights `[0.2912, 0.0106, 0.0982, 0.0625, 0.4917, 0.0458]`. Context vector shape `[6, 28]`.
4. **Reading the result.** Sorted vocabulary gives `Life=0, dessert=1, eat=2, first=3, is=4, short=5`, so the encoded sentence is `[0, 4, 5, 2, 1, 3]`. Row 1 is "is". It puts the most weight on "dessert" (0.49) and then "Life" (0.29). These weights come from random matrices, so they have no linguistic meaning yet. Training is what makes them meaningful.
5. **`randn` vs `rand`, the scale lesson.** With `randn` the raw scores were about ±100 and the softmax row was nearly one-hot: `[0.074, 0.924, 3.6e-18, 1.4e-3, 6.3e-18, 2.2e-8]`. With `rand` the maximum weight is 0.49. A saturated softmax has near-zero gradients on the non-winning entries, so it starves `Wq` and `Wk` of learning signal (the Sessions 1-2 saturation story).
6. **Why dividing by `sqrt(dk)` was not enough in the `randn` run.** The `sqrt(D)` argument assumes `q` and `k` have unit-variance components. Here both `x` and the weights are N(0, 1), so each `q` component has variance about C = 16, not 1. The score std becomes about `C * sqrt(D) = 16 * 4.9`, then `/ sqrt(D)` leaves a std of about 16. The scaling fixes only the `D` factor. A real model must also initialize the weights with std about `1/sqrt(C)` so that `q` and `k` start near unit variance. This feeds into Part 3.

**Open items for Session 11**

- Part 3: derive `Var(q·k) = D` and measure the four numbers.
- Part 4: shape ledger including the three loops.
- Parts 5-8: hand example, `loop_attention`, the six invariants, break-it log.

## Session 11 — Part 2: Concepts before code

The search-engine analogy: **query** is what a position is looking for, **key** is what each position advertises so it can be found, **value** is the payload delivered once a position is selected. It is intuition. It does not guarantee what a trained head learns.

**The three roles in detail.** Worked example: "The trophy didn't fit in the suitcase because **it** was too big."

| Role | Computed as | What it represents | Example |
|---|---|---|---|
| Query `q` | `X @ Wq` | What a specific token is actively looking for in the surrounding context. | For the pronoun "it", the network learns a query that hunts for singular, non-human nouns that appeared earlier. |
| Key `k` | `X @ Wk` | The metadata, or "advertisement", that a token broadcasts to the rest of the sequence. | For the earlier token "trophy", the key broadcasts that it is a singular, inanimate object. |
| Value `v` | `X @ Wv` | The actual information payload, the contextual meaning that is transferred if a match is made. | If "it" matches "trophy", what flows into "it" is the content of `v` for "trophy". |

- A match is a high `q · k` score. The query of "it" lines up with the key of "trophy".
- The key only decides *whether* "trophy" is picked. The value decides *what* is carried over. Both come from the same `x`, through different learned matrices.
- `X @ Wq` is the row-vector form used in this curriculum. The Raschka code writes the same thing as `embed @ W_query.T`, because his `W_query` is stored as `[dq, d]` (see the convention rewrite in Part 1).
- As with the search-engine analogy, this is a description of what training can make a head do. A random-weight head, like the one in my Part 1 run, does not do it.

**1. What question does a query represent computationally?**
A query is a vector in the same space as the keys. The question it asks is "which positions have keys that point in my direction?". It is answered by taking a dot product with every key. The result is one score per position, so the query produces a row of `T` scores (`q[i] · k[j]` for each `j`) and not an answer itself.

**2. What does a key do differently from a value?**
- A key is used only for **matching**. It decides *how much* attention a position receives, because it only enters the scores.
- A value is used only for **content**. It decides *what* flows into the output once the weights are fixed, because it only enters the weighted sum.
- They need separate vectors because "easy to find" and "useful to receive" are different jobs. The same position can have a key that says "I am a noun" and a value that carries what that noun means. This is also why `Dv` can differ from `D`: values never enter the dot product, so only `Dq == Dk` is required (see the run in Part 1: `dv = 28`, `dk = 24`).

**3. Why are query-key dot products scores rather than outputs?**
- A dot product is a single scalar per pair, shape `[T, T]`. It measures relevance but carries no content.
- It is unbounded and can be negative, so it cannot be used as a mixing weight directly. Softmax turns it into non-negative weights that sum to 1.
- The output has to live in value space, `[T, Dv]`. It is created by using the normalized scores to average the values.
- So the pipeline is score → normalize → mix. The scores are only the decision about where to look.

**Why not use the raw `x` for all three roles?**
If `q = k = v = x`, one vector has to be a good search term, a good index entry and a good payload at once. Two concrete problems:
- **The score matrix is symmetric.** `x_i · x_j = x_j · x_i`, so "i attends to j" always equals "j attends to i". Relations such as a verb looking for its subject are directional, and that cannot be expressed.
- **Self-match dominates.** For vectors of similar norm, `x_i · x_i = |x_i|²` is the largest score in the row, so every token mostly attends to itself and the output barely changes from the input.

With separate projections, `q[i] · k[j] = x_i Wq Wk^T x_j^T`, and `Wq Wk^T` is generally not symmetric, so directional relations become possible. Each projection can also be trained for its own job.

**Evidence from my run (random weights, no training):** in `attention_weights[1]`, token 1 ("is") gives itself only 0.0106 and gives "dessert" 0.4917. Without the projections, the self-score would be the biggest. This shows that the projections already break the self-match, though nothing has been learned yet.

## Session 11 — Part 3: The `sqrt(D)` scaling

Code: [8 session11.py](../../Month%204/8%20session11.py).

**Derivation.** Assume `q_x` and `k_x` are independent, mean 0, variance 1.

```text
Var(q_x k_x) = E[q_x^2] E[k_x^2] - (E[q_x] E[k_x])^2 = 1 * 1 - 0 = 1
Var(q . k)   = sum over x=1..D of Var(q_x k_x) = D      (independent terms, so variances add)
std(q . k)   = sqrt(D)
```

Each product has variance 1, but the dot product is a sum of `D` of them, so the variance grows linearly with `D`. Dividing by `sqrt(D)` divides the variance by `D` and returns it to 1.

**Measured** (`torch.randn(10000, 20)` for `q` and `k`, so `D = 20`, not the 64 in the curriculum; seed 10):

| Quantity | Expected | Measured |
|---|---|---|
| `q.var()`, `k.var()` | 1 | 1.0008, 1.0030 |
| `(q * k).var()` (per-element product) | 1 | 1.0027 |
| `(q * k).sum(-1).var()` (dot product) | `D` = 20 | **20.07** |
| `((q * k) / sqrt(20)).sum(-1).var()` | 1 | **1.0035** |

Softmax saturation (one row of 10 scores, `a = torch.randn(1, 10)`, seed 10, `a.var() = 1.035`):

| Scores | Max probability | Other entries |
|---|---|---|
| `a` | 0.5175 | spread, 0.02 to 0.19 |
| `a * 10` (std about 10) | **0.99995** | about 1e-10 to 1e-13 |
| `a * 10 / sqrt(10)` | 0.955 | still saturated, std about 3.2 |
| `a * 10 / (a * 10).std()` | 0.509 | back to nearly the original |

The third row is a wrong divisor, not a failure of the idea. The scores were scaled by 10, so their std is 10, not `sqrt(10)`. The divisor must equal the std of the scores. For real dot products the std is `sqrt(D)`, which is why `/sqrt(D)` is the right constant there.

**Link to Sessions 1-2.** The softmax Jacobian is `dp_i/ds_j = p_i (delta_ij - p_j)`. When `p` is nearly one-hot, every entry is near 0, so almost no gradient flows back to the scores and none reaches `Wq` and `Wk`. Unscaled scores at large `D` land in that regime, and the heads cannot learn. After scaling, the softmax is **not saturated**: there is still a clear preference, but the gradients are alive. It is not "linear", because the softmax is still strongly non-linear.

**Caveat from the Part 1 run.** `/sqrt(D)` is a fixed constant standing in for "divide by the true std of the scores". It is exact only if `q` and `k` have unit variance and are independent. In the `torch.rand` run, `query.var()` is 3.4 and `key.var()` is 3.7, `q` and `k` are correlated (same `x`, all-positive weights), and `normalized_omega.var()` is still 137, not 1. Fixing that needs an init with weight std about `1/sqrt(C)`, not a different divisor. Dividing by the measured score std (last row above) would also work, but attention does not do that.

**Not done yet.** The curriculum's exact numbers (`D = 64`, scale by 8, `softmax(randn(10) * 8)`) were not run. I ran `D = 20` and a `* 10` scale, and the pattern is the same.

## Session 11 — Part 4: Shape ledger

Notation: `T` = tokens, `d` = embedding size (`C` in the curriculum), `dq = dk`, `dv`. In the Part 1 run: `T = 6`, `d = 16`, `dq = dk = 24`, `dv = 28`. The run is unbatched, so the curriculum's `[B, T, ...]` shapes are these with `B` in front.

| Tensor | Symbolic | Run | How it is made |
|---|---|---|---|
| `X` (embed) | `[T, d]` | `[6, 16]` | `embed_layer(encoded)` |
| `W_query`, `W_key` | `[dk, d]` | `[24, 16]` | `torch.rand` |
| `W_value` | `[dv, d]` | `[28, 16]` | `torch.rand` |
| `Q`, `K` | `[T, dk]` | `[6, 24]` | `X @ W.T` |
| `V` | `[T, dv]` | `[6, 28]` | `X @ W_value.T` |
| `omega`, `scaled_omega` | `[T, T]` | `[6, 6]` | `Q @ K.T`, then `/ sqrt(dk)` |
| `weights` | `[T, T]` (batched: `[B, T, T]`) | `[6, 6]` | `softmax(scaled_omega, dim=1)` |
| `out` (context) | `[T, dv]` (batched: `[B, T, Dv]`) | `[6, 28]` | `weights @ V` |

**Correction I made while writing this.** `omega` is `[T, T]`, not `[d, d]`. It is 6×6 because there are 6 tokens, not because of the embedding size (that is 16, so a `[d, d]` matrix would be 16×16). Entry `(i, j)` is how much position `i` attends to position `j`.

**Other points**
- Softmax uses `dim=1` (the key axis `j`). Each row sums to 1.
- The scaling divisor is `sqrt(dk)` (24), not `sqrt(d)`.
- `dv != dk` is fine, because only `dq == dk` is needed for `Q @ K.T`.

**The three loops** (the file has none; the matmuls do the looping. They appear in `loop_attention`.)

| Loop | Axis it walks | One iteration means |
|---|---|---|
| `i` | query rows of `Q`, rows of `weights` and `out` | which position is asking |
| `j` | key rows of `K`, columns of `omega`, rows of `V` | which position is being looked at |
| `x` | the `dk` axis, in `Q[i] . K[j]` | one feature of the dot product (sum of `dk` terms) |

For the output, `out[i] = sum over j of weights[i, j] * V[j]`. Here `j` is again the position being looked at, and the vector `V[j]` has `dv` entries.

## Session 11 — Part 5: Three-token example by hand

Code: the Part 5 cell in [8 session11.py](../../Month%204/8%20session11.py).

**Setup.** `B=1`, `T=3`, `C=D=Dv=2`, `Wq=Wk=Wv=I`, so `q = k = v = x`. Identity projections remove the projection step, which leaves only score → softmax → mix.

```text
x0 = [1, 0]    x1 = [0, 1]    x2 = [1, 1]
```

**Causal rule.** Position `i` may only look at keys `j <= i`. In the code this is `range(i + 1)`, so disallowed keys are never scored. No mask tensor is needed in the loop version.

**Recipe per query row `i`** (each step is one line of the code)

| Step | Code | Meaning |
|---|---|---|
| Products | `q[i] * k[j]` for `j` in `range(i+1)`, stacked | Elementwise products, shape `[i+1, D]`. This is the dot product before the sum. |
| Dot product | `.sum(dim=-1)` | One score per allowed key, shape `[i+1]`. |
| Scale | `/ D**0.5` | Divide by `sqrt(2) = 1.4142`. |
| Softmax | `torch.softmax(scaled, dim=0)` | `dim=0` here because the scores are a 1-D vector of allowed keys. It is the key axis, like `dim=-1` in the `[T, T]` matrix. |
| Mix | `weights[j] * v[j]`, stacked, `.sum(dim=0)` | Weighted sum of the allowed value vectors, shape `[D]`. |

**Results**

| `i` | Allowed keys | Dot products | Scaled | Weights | Output |
|---:|---|---|---|---|---|
| 0 | 0 | `[1]` | `[0.7071]` | `[1]` | `[1, 0]` |
| 1 | 0, 1 | `[0, 1]` | `[0, 0.7071]` | `[0.3302, 0.6698]` | `[0.3302, 0.6698]` |
| 2 | 0, 1, 2 | `[1, 1, 2]` | `[0.7071, 0.7071, 1.4142]` | `[0.2483, 0.2483, 0.5035]` | `[0.7517, 0.7517]` |

These match the curriculum's answer table.

**Worked row 2**, as a check on the arithmetic: `exp([0.7071, 0.7071, 1.4142]) = [2.028, 2.028, 4.113]`, sum `8.169`, so weights `[0.2483, 0.2483, 0.5035]`. Output = `0.2483*[1,0] + 0.2483*[0,1] + 0.5035*[1,1] = [0.7517, 0.7517]`.

**What to notice**
- Every row of weights sums to 1.
- Token 0 has nothing earlier to look at, so it copies itself (`weights = [1]`, `out = v0`).
- Token 2 matches itself best (score 2 against 1 and 1), so it gets the largest weight, 0.50.
- Each output is a mix of earlier value vectors only. That is what causal attention guarantees.
- The `stacked_scores` print shows the products before the sum. This makes it visible that the dot product is "multiply per feature, then sum over the `D` axis", which is the `x` loop from Part 4.

**Not done in this cell.** There is no vectorized (`masked_fill` with `-inf`) version and no `assert` against the table yet. The check against the curriculum answers is by reading the printed values. Those are for the `loop_attention` function and the six invariants in Parts 6 to 8.

## Session 11 — Part 6: `loop_attention`

Code: `loop_attention` in [8 session11.py](../../Month%204/8%20session11.py). Shapes: `x [B, T, D]`, `Wq/Wk [dq, D]`, `Wv [dv, D]`, returns `(out [B, T, dv], weights [B, T, T])`. (Weights are stored `[out, in]` like Raschka, so the projection is `x @ W.T`.)

**What the function does, in order**

| Stage | Loops | What happens |
|---|---|---|
| Project | `b, i, j, k` | `Q[b,i,j] += x[b,i,k] * Wq.T[k,j]`, likewise `K`, `V`. Each is asserted against `x @ W.T`. Rule 1 allows this to be vectorized; I wrote it out to see the indices. |
| Score | `b, i, j in range(i+1)` | `scores[b,i,j] = (Q[b,i] * K[b,j]).sum() / D**0.5`. The array starts as zeros and only the lower triangle is written. |
| Softmax | `b, i` | `row = scores[b,i,:i+1]` (permitted scores only), `exp(row - row.max())`, divide by the sum, write into `weights[b,i,:i+1]`. Above the diagonal `weights` is never written, so it stays exactly 0. |
| Aggregate | `b, i, j in range(i+1)` | `acc = acc + weights[b,i,j] * V[b,j]`, then `out[b,i] = acc`. Asserted against `weights @ V`. |

**Intuition for the aggregation.** Q and K only decided *where* to look. `V[b, j]` is the content token `j` offers, and `weights[b, i, j]` is how much of it token `i` takes. `out[b, i]` is a weighted average of the visible value vectors. Small example (`T=3`, `dv=5`), weights row 2 = `[0.2, 0.3, 0.5]`:

```text
out[2]    = 0.2*V[0] + 0.3*V[1] + 0.5*V[2]
out[2, 3] = 0.2*V[0,3] + 0.3*V[1,3] + 0.5*V[2,3]      (one output dimension)
```

Token 0 sees only itself (`weights = [1, 0, 0]`), so `out[0] == V[0]`. `weights @ V` is a batched matmul: for each `b`, `[T, T] @ [T, dv] -> [T, dv]`, and the inner `T` is the `j` sum.

**Bugs I hit on the way**
- Starting `scores` as `torch.zeros` and calling `torch.softmax(scores, dim=2)` gives future positions weight, because `exp(0) = 1`. A masked softmax needs `-inf` (`exp(-inf) = 0`). The manual per-row softmax avoids this because the future is never in the slice. The `-inf` version is kept as a commented-out block in the file.
- Using `+=` on `acc` risks in-place problems with autograd. `acc = acc + ...` is safe.

**Checks run** (a scratch script that executes the function from the file; shapes `B=2, T=5, C=6, D=4, Dv=3`)

| Check | Result |
|---|---|
| Output shapes | `out [2, 5, 3]`, `weights [2, 5, 5]` |
| Rows sum to 1 | all 1.0000 |
| Above the diagonal | `torch.triu(weights, 1).abs().max() = 0.0` |
| `out.sum().backward()` | runs, `Wq.grad` populated (abs sum 30.84) |
| Part 5 example (`Wq=Wk=Wv=I`) | weights `[1]`, `[0.3302, 0.6698]`, `[0.2483, 0.2483, 0.5035]`; out `[1, 0]`, `[0.3302, 0.6698]`, `[0.7517, 0.7517]`. Matches the table. |

**Known gap: the scale uses `D`, not `dk`.** The file divides by `D ** 0.5` where `D = x.shape[-1]` is the *input* dimension. The derivation in Part 3 says the divisor is `sqrt` of the dimension that the dot product sums over, which is `dk`. In the Part 5 example `D = dk = 2`, and the table matches, but with `D=4, dk=5` the scores are scaled by the wrong constant. The fix is `Q.shape[-1] ** 0.5`. I have not changed it yet.

## Session 11 — First-principles questions 4 to 7, in terms of `loop_attention`

Questions 1 to 3 are answered in Part 2.

**4. Why does softmax operate over the key-position axis?**
Row `i` of the weights is the answer to "how does query `i` spend its attention over the keys it can see?". That is a distribution over `j`, so softmax normalizes along `j`. In the code it is `row = scores[b, i, :i+1]`, one softmax per query. This makes `out[b, i] = sum_j w_j * V[b, j]` a weighted average with weights summing to 1, so the output stays on the scale of `V`. Softmax over the query axis would normalize *columns*: it would say how key `j` distributes itself across queries, and the rows of `weights` would no longer sum to 1, so the mixture would not be an average.

**5. Why is the causal mask applied before softmax?**
Softmax divides by the sum over the whole row, so anything in that row changes every other weight. If the future scores were included and then zeroed afterward, the surviving past weights would be shrunk by the future's share of the denominator, and the row would sum to less than 1. Worse, the future tokens would influence the past through that denominator, so changing `x[:, i+1:]` would change `out[:, :i+1]` (invariant 3 would fail). Masking before softmax makes the future contribute exactly `exp(-inf) = 0` to the max and the sum. `loop_attention` gets the same effect by never putting the future in the slice (`range(i + 1)` and `:i+1`). The test cell with `-inf` also shows it: `softmax([10, -inf, -inf]) = [1, 0, 0]`.

**6. Why does dividing by `sqrt(D)` help keep score scale controlled?**
`Var(q . k) = dk` for unit-variance independent components, so the std of a raw score grows as `sqrt(dk)`. At large `dk` the softmax saturates to one-hot, and the gradient to `Wq` and `Wk` vanishes (Part 3: measured variance 20.07, then 1.0035 after scaling). Dividing by `sqrt(dk)` returns the variance to 1. In my function this line is `/ D ** 0.5`, which is the gap noted in Part 6: it should use `dk`. It also only holds if `q` and `k` start near unit variance, which is an initialization job (weight std about `1/sqrt(C)`).

**7. Time and memory complexity in sequence length?**

| Part | Count | Cost |
|---|---|---|
| Scores | `B * T(T+1)/2` pairs, each a dot product over `D` | `O(B·T²·D)` |
| Aggregation | `B * T(T+1)/2` pairs, each scales a `dv` vector | `O(B·T²·dv)`, the same order as scores |
| Projections | `B·T` rows times `C` times `(dq + dk + dv)` | `O(B·T·C·D)`, linear in `T` |
| Stored `weights` | `B * T * T` floats | `O(B·T²)` |

- Time is `O(B·T²·D)`: `B` for the batch, `T²` for every query paired with every key, `D` for the length of each dot product. The causal triangle halves the pairs, and that constant is dropped.
- Memory for the weights is `O(B·T²)`: one number per query-key pair, per batch. `D` does not appear because a weight is a scalar. `Q`, `K`, `V` are only `O(B·T·D)`.
- The loop version also keeps `scores` as a second `[B, T, T]` tensor, so it uses twice the weight memory, which is still `O(B·T²)`.
- Doubling `T` quadruples both the time and the weight memory. This is the cost that later sessions (KV cache, FlashAttention) work around.
- The loop version runs about `B·T²/2` Python iterations, plus the 4-deep projection loops, so it is far slower than the vectorized one. It is for checking, not for use.

**Not done yet.** Part 7 (`tests/test_attention_forward.py` with the six invariants and the fixed Part 5 regression test), Part 8 (break-it log), and the `dk` scale fix.
